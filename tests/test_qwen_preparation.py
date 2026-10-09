import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import prepare_qwen as q


def cast():
    return {"characters": [
        {"id": "narrator", "name": "Автор", "gender": "male", "age": "adult", "aliases": [],
         "evidence": "Повествование", "voice_description": "Warm adult male baritone, measured rhythm."},
        {"id": "anna", "name": "Анна", "gender": "female", "age": "adult", "aliases": ["Аня"],
         "evidence": "сказала Анна", "voice_description": "Bright female alto, brisk and precise articulation."},
    ]}


class PreparationTests(unittest.TestCase):
    def test_roles_are_lossless_and_narrator_insert_is_separate(self):
        batch = [{"id": "p1", "text": "— Да, — сказала Анна, — пойду."}]
        answer = {"paragraphs": [{"id": "p1", "segments": [
            {"text": "— Да,", "speaker": "anna", "confidence": "high"},
            {"text": "— сказала Анна, —", "speaker": "narrator", "confidence": "high"},
            {"text": "пойду.", "speaker": "anna", "confidence": "high"}]}]}
        labels = q.validate_roles(answer, batch, cast())
        self.assertEqual([s["speaker"] for s in labels[0]["spans"]], ["anna", "narrator", "anna"])
        answer["paragraphs"][0]["segments"][2]["text"] = "уйду."
        with self.assertRaises(ValueError):
            q.validate_roles(answer, batch, cast())

    def test_reject_missing_unknown_and_truncated_roles(self):
        batch = [{"id": "p1", "text": "Привет. Пока."}]
        for segments in ([{"text": "Привет.", "speaker": "anna", "confidence": "high"}],
                         [{"text": "Привет. Пока.", "speaker": "ghost", "confidence": "high"}],
                         [{"text": "Привет. Пока.", "speaker": None, "confidence": "low"}]):
            with self.assertRaises(ValueError):
                q.validate_roles({"paragraphs": [{"id": "p1", "segments": segments}]}, batch, cast())
        with self.assertRaises(ValueError):
            q.parse_reply({"done_reason": "length", "message": {"content": "{}"}})

    def test_reporting_insert_cannot_be_read_by_character(self):
        batch = [{"id": "p1", "text": "– Конечно, – тихо сказала Анна."}]
        for speaker in ("anna", "narrator"):
            answer = {"paragraphs": [{"id": "p1", "segments": [
                {"text": batch[0]["text"], "speaker": speaker, "confidence": "high"}]}]}
            with self.assertRaises(ValueError):
                q.validate_roles(answer, batch, cast())
        # A reporting verb spoken as part of a character's own sentence is allowed.
        batch = [{"id": "p1", "text": "– Он сказал мне правду."}]
        q.validate_roles({"paragraphs": [{"id": "p1", "segments": [
            {"text": batch[0]["text"], "speaker": "anna", "confidence": "high"}]}]}, batch, cast())

    def test_truncated_reply_increases_budget_in_file_workflow(self):
        args = SimpleNamespace(command="advance", model="test", endpoint="https://ollama.com", no_think=False, max_tokens=16000)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            q.obtain(out, "cast", "prompt", q.validate_cast, args, None)
            pending = q.load(out / "pending.json")
            q.write(pending["response"], {"done_reason": "length", "message": {"content": "{}"}})
            self.assertIsNone(q.obtain(out, "cast", "prompt", q.validate_cast, args, None))
            request = q.load(q.load(out / "pending.json")["request"])
            self.assertEqual(request["options"]["num_predict"], 32000)

    def test_cast_does_not_share_voices(self):
        value = cast()
        q.validate_cast(value)
        value["characters"][1]["voice_description"] = value["characters"][0]["voice_description"]
        with self.assertRaises(ValueError):
            q.validate_cast(value)

    def test_short_chunks_preserve_every_character_and_words(self):
        text = ("Это достаточно длинное предложение, чтобы проверить нарезку. " * 40).strip()
        cuts = list(q.split_text(text, 140, 180))
        self.assertEqual("".join(text[a:b] for a, b, _ in cuts), text)
        self.assertTrue(all(b - a <= 180 for a, b, _ in cuts))
        self.assertTrue(all(boundary in {"sentence", "end"} for _, _, boundary in cuts))
        long_sentence = ("слово " * 100).strip()
        self.assertTrue(any(boundary == "word" for _, _, boundary in q.split_text(long_sentence, 100, 130)))
        with self.assertRaises(ValueError):
            list(q.split_text("а" * 500, 100, 130))

    def test_epub_inline_markup_accents_and_spine_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.epub"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
                z.writestr("OEBPS/book.opf", '<package xmlns="http://www.idpf.org/2007/opf"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Тест</dc:title></metadata><manifest><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/><item id="b" href="b.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="b"/><itemref idref="a"/></spine></package>')
                z.writestr("OEBPS/a.xhtml", '<html><body><p id="a">Про<em>ве</em>рка мы́сли.</p></body></html>')
                z.writestr("OEBPS/b.xhtml", '<html><body><h1>Глава</h1><blockquote><p>Первый абзац.</p></blockquote></body></html>')
            source = q.read_source(path)
            self.assertEqual([p["text"] for p in source["paragraphs"]], ["Глава", "Первый абзац.", "Проверка мы́сли."])
            self.assertEqual(source["paragraphs"][-1]["source"]["element_id"], "a")

    def test_export_and_check_detect_loss_and_plan_reordering(self):
        paragraph = {"id": "p1", "chapter_id": "ch001", "text": "Тихая улица. " * 50,
                     "source": {"href": "chapter.xhtml", "block_index": 0, "element_id": None, "tag": "p"}}
        source = {"title": "Тест", "author": "", "input_sha256": "hash", "normalization": "test",
                  "paragraphs": [paragraph], "chapters": [{"id": "ch001", "href": "chapter.xhtml", "title": "Глава"}]}
        labels = [{"id": "p1", "spans": [{"start": 0, "end": len(paragraph["text"]), "speaker": "anna", "confidence": "high", "reason": ""}]}]
        args = SimpleNamespace(target_chars=100, max_chars=140, model="test", no_think=False)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            q.export_project(out, source, cast(), labels, args)
            manifest = q.load(out / "manifest.json")
            q.check_manifest(manifest, source)
            manifest["chapters"][0]["segment_ids"].reverse()
            with self.assertRaises(ValueError):
                q.check_manifest(manifest, source)
            manifest = q.load(out / "manifest.json")
            manifest["segments"].pop()
            with self.assertRaises(ValueError):
                q.check_manifest(manifest, source)

    def test_file_workflow_is_cached_and_model_specific(self):
        args = SimpleNamespace(command="advance", model="test", endpoint="https://ollama.com", no_think=False, max_tokens=16000)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.assertIsNone(q.obtain(out, "cast", "prompt", q.validate_cast, args, None))
            pending = q.load(out / "pending.json")
            q.write(pending["response"], {"done_reason": "stop", "message": {"content": json.dumps(cast())}})
            self.assertEqual(q.obtain(out, "cast", "prompt", q.validate_cast, args, None), cast())
            args.model = "different"
            self.assertIsNone(q.obtain(out, "cast", "prompt", q.validate_cast, args, None))


if __name__ == "__main__":
    unittest.main()
