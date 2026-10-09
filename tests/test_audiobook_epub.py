import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET

import audiobook_epub as a


def fixture():
    segments = []
    for i, (paragraph, chapter, text, speaker) in enumerate([
        ("p1", "ch1", "Привет, ", "narrator"), ("p1", "ch1", "Арина!", "arina"),
        ("p2", "ch2", "Я здесь. Ёж и ещё один ёж.", "arina")]):
        segments.append({"id": f"s{i}", "order": i, "paragraph_id": paragraph, "chapter_id": chapter,
            "text": text, "speaker": speaker, "source": {"href": "original.xhtml", "block_index": i},
            "pause_after_ms": 100, "needs_review": False})
    return {"book": {"title": "Тест & книга", "author": "Автор", "input_sha256": "original"},
        "segments": segments, "voices": [{"id": "narrator"}, {"id": "arina"}],
        "chapters": [{"id": "ch1", "title": "Первый раздел"}, {"id": "ch2", "title": "Второй раздел"}]}


def timeline(manifest):
    return [{"segment_id": s["id"], "speaker": s["speaker"], "text": s["text"], "source": s["source"],
             "audio_start": i * 1.1, "audio_end": i * 1.1 + 1} for i, s in enumerate(manifest["segments"])]


class BookTests(unittest.TestCase):
    def test_closing_dialogue_dash_is_not_a_separate_tts_sentence(self):
        m = fixture()
        row = m["segments"][0]
        text = "– Вот и спасибо! –"
        row.update(text=text)
        row["source"].update(char_start=0, char_end=len(text))
        m["segments"] = [row]
        m["paragraph_texts"] = {"p1": text}
        split = a.sentence_segments(m)
        self.assertEqual(len(split), 1)
        self.assertEqual(split[0]["text"], text)
        self.assertEqual(split[0]["speaker"], row["speaker"])
        self.assertEqual(split[0]["source"], row["source"])

    def test_sentence_mode_preserves_roles_offsets_and_paragraph(self):
        manifest = fixture()
        text = "И. И. Иванов пришёл. Арина ответила: «Да!» Потом стало тихо."
        row = manifest["segments"][0]
        row.update(text=text, pause_after_ms=300)
        row["source"].update(char_start=0, char_end=len(text))
        manifest["segments"] = [row]
        manifest["paragraph_texts"] = {"p1": text}
        manifest["segments"] = a.sentence_segments(manifest)
        self.assertEqual(len(manifest["segments"]), 3)
        self.assertEqual(manifest["segments"][0]["text"], "И. И. Иванов пришёл.")
        self.assertTrue(all(s["speaker"] == "narrator" for s in manifest["segments"]))
        self.assertEqual([s["pause_after_ms"] for s in manifest["segments"]], [450, 600, 800])
        with tempfile.TemporaryDirectory() as folder:
            a.build_text_epub(manifest, Path(folder) / "sentence.epub")
            a.validate_epub(Path(folder) / "sentence.epub", manifest)

    def test_spaces_trimmed_for_tts_are_preserved_in_epub(self):
        manifest = fixture()
        manifest["segments"] = manifest["segments"][:2]
        manifest["segments"][0]["text"] = "Привет,"
        manifest["segments"][0]["source"].update(char_start=0, char_end=8)
        manifest["segments"][1]["source"].update(char_start=8, char_end=14)
        manifest["paragraph_texts"] = {"p1": "Привет, Арина!"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "book.epub"
            a.build_text_epub(manifest, path)
            with zipfile.ZipFile(path) as archive:
                root = ET.fromstring(archive.read("EPUB/ch1.xhtml"))
                self.assertEqual("".join(root.find(".//{" + a.XHTML + "}p").itertext()), "Привет, Арина!")

    def test_excerpt_never_cuts_paragraph(self):
        manifest = fixture()
        rows, duration = a.select_excerpt(manifest, timeline(manifest), 2.0, "p1")
        self.assertEqual([s["id"] for s in rows], ["s0", "s1"])
        self.assertAlmostEqual(duration, 2.2)
        with self.assertRaises(ValueError):
            a.select_excerpt(manifest, timeline(manifest)[:1], 2, "p1")

    def test_epub_preserves_text_anchors_and_original_sources(self):
        manifest = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "book.epub"
            a.build_text_epub(manifest, path)
            self.assertEqual(a.validate_epub(path, manifest)["text_segments"], 3)
            self.assertEqual(manifest["segments"][0]["original_source"]["href"], "original.xhtml")
            with zipfile.ZipFile(path) as archive:
                doc = ET.fromstring(archive.read("EPUB/ch1.xhtml"))
                p = doc.find(".//{" + a.XHTML + "}p")
                self.assertEqual("".join(p.itertext()), "Привет, Арина!")

    def test_incomplete_or_wrong_timeline_rejected(self):
        manifest = fixture()
        rows = timeline(manifest)
        a.verify_timeline(manifest, rows, 3.3)
        with self.assertRaises(ValueError):
            a.verify_timeline(manifest, rows[:-1], 3.3)
        rows[0]["speaker"] = "arina"
        with self.assertRaises(ValueError):
            a.verify_timeline(manifest, rows, 3.3)

    @unittest.skipUnless(Path(a.FFMPEG).exists(), "FFmpeg required")
    def test_real_m4b_and_media_overlays_packaging(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fixture()
            a.build_text_epub(manifest, root / "text.epub")
            a.save(root / "manifest.json", manifest)
            audio = root / "audio"
            audio.mkdir()
            sf.write(audio / "sample.wav", np.sin(np.arange(79200) * 0.08).astype("float32") * 0.1, 24000)
            a.save(audio / "timeline.json", {"segments": timeline(manifest)})
            a.save(audio / "benchmark.json", {"state": "complete", "final_audio_seconds": 3.3, "book": manifest["book"]})
            a.package(root / "manifest.json", audio, root / "result")
            result = a.load(root / "result/package-report.json")
            self.assertEqual(result["chapters"], 2)
            self.assertEqual(result["validation"]["synchronized_segments"], 3)
            with zipfile.ZipFile(root / "result/book-read-along.epub") as archive:
                self.assertEqual(archive.read("EPUB/audio/book.m4a"), (root / "result/book.m4b").read_bytes())
                smil = ET.fromstring(archive.read("EPUB/ch1.smil"))
                clips = smil.findall(".//{" + a.SMIL + "}audio")
                self.assertEqual(clips[0].get("clipEnd"), "1.100000s")
                chapter = ET.fromstring(archive.read("EPUB/ch1.xhtml"))
                self.assertEqual(chapter.get("lang"), "ru")
                self.assertTrue(any(e.get("href") == "style.css" for e in chapter.findall(".//{" + a.XHTML + "}link")))
                original_audio = archive.read("EPUB/audio/book.m4a")
            repaired = root / "repaired.epub"
            a.repair_highlighting(root / "result/book-read-along.epub", repaired)
            a.validate_epub(repaired, manifest, overlays=True)
            with zipfile.ZipFile(repaired) as archive:
                self.assertEqual(archive.read("EPUB/audio/book.m4a"), original_audio)


if __name__ == "__main__":
    unittest.main()
