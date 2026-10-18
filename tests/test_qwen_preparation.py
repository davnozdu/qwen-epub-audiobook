import json
import pathlib
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

    def test_short_pieces_are_glued_to_the_same_voice_and_never_mix_voices(self):
        """Короткие куски одного голоса склеиваются — и внутри абзаца, и через целые абзацы.

        Реплики по одному-два слова модель в одиночку рвёт; их надо отдавать одной порцией.
        Чужие голоса не смешиваются никогда: сегмент озвучивается одним голосом, и склейка
        разных голосов сломала бы и звук, и разметку.
        """
        long_narrator = "Дальше идёт длинная реплика рассказчика, которую резать незачем. " * 6
        source = {"title": "Тест", "author": "", "input_sha256": "hash", "normalization": "test",
                  "paragraphs": [
                      {"id": "p1", "chapter_id": "ch001", "text": "— Да.",
                       "source": {"href": "a.xhtml", "block_index": 0, "element_id": None, "tag": "p"}},
                      {"id": "p2", "chapter_id": "ch001", "text": "— Нет.",
                       "source": {"href": "a.xhtml", "block_index": 1, "element_id": None, "tag": "p"}},
                      {"id": "p3", "chapter_id": "ch001", "text": "— Не знаю.",
                       "source": {"href": "a.xhtml", "block_index": 2, "element_id": None, "tag": "p"}},
                      {"id": "p4", "chapter_id": "ch001", "text": long_narrator,
                       "source": {"href": "a.xhtml", "block_index": 3, "element_id": None, "tag": "p"}}],
                  "chapters": [{"id": "ch001", "href": "a.xhtml", "title": "Глава"}]}
        labels = [{"id": p["id"], "spans": [{"start": 0, "end": len(p["text"]), "speaker": "narrator",
                                            "confidence": "high", "reason": ""}]}
                  for p in source["paragraphs"]]
        args = SimpleNamespace(target_chars=220, max_chars=360, min_chars=80, model="test", no_think=False)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            q.export_project(out, source, cast(), labels, args)
            manifest = q.load(out / "manifest.json")
            q.check_manifest(manifest, source)
            segments = manifest["segments"]
        # Три коротких абзаца подряд ушли одной порцией вместе с началом длинного абзаца:
        # короткий кусок модель в одиночку рвёт, поэтому к нему дописан сосед того же голоса.
        self.assertGreaterEqual(len(segments), 2, [s["text"] for s in segments])
        self.assertTrue(segments[0]["text"].startswith("— Да. — Нет. — Не знаю."), segments[0]["text"])
        self.assertEqual([piece["paragraph_id"] for piece in segments[0]["sources"]], ["p1", "p2", "p3", "p4"])
        self.assertEqual([piece["text"] for piece in segments[0]["sources"]][:3],
                         ["— Да.", "— Нет.", "— Не знаю."])
        self.assertEqual([s["order"] for s in segments], list(range(len(segments))))
        # Потолка не превышаем: сегмент не длиннее максимума, иначе модель начнёт сбиваться.
        self.assertTrue(all(len(s["text"]) <= 360 for s in segments), [len(s["text"]) for s in segments])
        # Ни одна строка книги не потеряна и порядок не изменён.
        self.assertEqual(" ".join(s["text"] for s in segments),
                         " ".join(p["text"] for p in source["paragraphs"]).strip())

    def test_short_tail_inside_one_paragraph_is_glued_to_its_neighbour(self):
        """Короткий хвост абзаца приклеивается к предыдущему куску, а не идёт отдельной порцией."""
        text = "Тихая улица, и никого. " * 20 + "И всё."
        source = {"title": "Тест", "author": "", "input_sha256": "hash", "normalization": "test",
                  "paragraphs": [{"id": "p1", "chapter_id": "ch001", "text": text.strip(),
                                  "source": {"href": "a.xhtml", "block_index": 0, "element_id": None, "tag": "p"}}],
                  "chapters": [{"id": "ch001", "href": "a.xhtml", "title": "Глава"}]}
        labels = [{"id": "p1", "spans": [{"start": 0, "end": len(text.strip()), "speaker": "narrator",
                                         "confidence": "high", "reason": ""}]}]
        args = SimpleNamespace(target_chars=220, max_chars=360, min_chars=80, model="test", no_think=False)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            q.export_project(out, source, cast(), labels, args)
            manifest = q.load(out / "manifest.json")
            q.check_manifest(manifest, source)
            segments = manifest["segments"]
        self.assertTrue(all(len(s["text"]) >= 80 for s in segments), [len(s["text"]) for s in segments])
        self.assertEqual(" ".join(s["text"] for s in segments), text.strip())

    def test_defaults_keep_a_segment_inside_the_range_the_model_handles(self):
        """Пороги по умолчанию укладываются в 15-20 секунд речи.

        Замеры по готовым книгам: 12-13,4 знака в секунду, в среднем 12,5. Цель нарезки и потолок
        должны попадать в те самые 15-20 секунд, иначе модель либо рвётся на коротком, либо
        сбивается на длинном.
        """
        import argparse, contextlib, io
        rate = 12.5                      # знаков в секунду, замерено на четырёх книгах
        parser = argparse.ArgumentParser()
        defaults = {}
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                q.main()             # без аргументов argparse печатает справку и выходит
            except SystemExit:
                pass
        source = pathlib.Path(q.__file__).read_text(encoding="utf-8")
        for name, seconds in (("--target-chars", (15, 20)), ("--max-chars", (15, 21)),
                              ("--min-chars", (5, 9))):
            line = next(line for line in source.splitlines() if f'"{name}"' in line)
            value = int(line.split("default=")[1].split(",")[0])
            defaults[name] = value
            low, high = seconds
            self.assertLessEqual(low, value / rate, f"{name}={value} короче {low} с")
            self.assertLessEqual(value / rate, high, f"{name}={value} длиннее {high} с")
        self.assertLess(defaults["--min-chars"], defaults["--target-chars"])
        self.assertLess(defaults["--target-chars"], defaults["--max-chars"])

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
        args = SimpleNamespace(target_chars=100, max_chars=140, min_chars=80, model="test",
                               no_think=False)
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
