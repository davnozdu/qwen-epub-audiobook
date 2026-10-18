"""Сборка книги из частей: склейка, сверка стыка, громкость и музыка на всю книгу.

Тест гоняет НАСТОЯЩИЙ glue_book.py на двух частях, собранных теми же шагами, что и на машине
(манифест → prepare → синтез), с настоящими ffmpeg/ffprobe. Синтез подделан: вместо речи —
короткие тоны, потому что GPU в тесте нет, — а всё остальное (сырьё во FLAC, timeline,
benchmark, склейка, normalize, музыка, сборка m4b и read-along) настоящее.

    python3 tests/test_glue_book.py
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile
from contextlib import contextmanager

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "colab"))

import audiobook_epub as a                                    # noqa: E402

WORK = pathlib.Path(tempfile.mkdtemp(prefix="glue-test-"))

CHAPTERS = [("Первый раздел", ["Арина вошла в комнату и закрыла дверь.",
                               "Снаружи шумел дождь, и было слышно, как капает с крыши."]),
            ("Второй раздел", ["Она села у окна и открыла книгу.",
                               "Страницы были старые, но буквы ещё читались."])]


def make_epub(path, title, chapters) -> pathlib.Path:
    """Книга как её видит конвейер: главы документами, абзацы — тегами <p>."""
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier("urn:test:" + title)
    book.set_title(title)
    book.set_language("ru")
    docs = []
    for index, (chapter_title, paragraphs) in enumerate(chapters, 1):
        doc = epub.EpubHtml(uid=f"ch{index:03d}", file_name=f"ch{index:03d}.xhtml",
                            title=chapter_title, lang="ru")
        doc.content = ("<html><body>"
                       + "".join(f'<p id="p{index:03d}{n + 1:03d}">{text}</p>'
                                 for n, text in enumerate(paragraphs))
                       + "</body></html>")
        book.add_item(doc)
        docs.append(doc)
    book.toc = tuple(docs)
    book.spine = docs
    epub.write_epub(str(path), book)
    return path


def fake_synthesis(part) -> None:
    """Синтез на машине: сегменты, timeline, benchmark. Потом всё сырьё уезжает во FLAC."""
    source = json.loads((part / "project" / "source" / "manifest.json").read_text(encoding="utf-8"))
    audio = part / "project" / "audio"
    (audio / "segments").mkdir(parents=True, exist_ok=True)
    rate, cursor, rows, spoken = 24000, 0.0, [], {}
    for index, segment in enumerate(source["segments"]):
        seconds = max(0.25, min(2.0, len(segment["text"]) / 14))
        begin, end = round(cursor * rate) / rate, round((cursor + seconds) * rate) / rate
        name = f"{segment['id']}.wav"
        with wave.open(str(audio / "segments" / name), "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            frames = int((end - begin) * rate)
            handle.writeframes(b"".join(int(3000 * (index % 3 + 1) * ((i // 40) % 2 * 2 - 1))
                                        .to_bytes(2, "little", signed=True) for i in range(frames)))
        # Ударения и числа: манифест части хранит то, что реально произносилось.
        spoken[segment["id"]] = segment["text"]
        rows.append({"segment_id": segment["id"], "speaker": segment["speaker"], "text": segment["text"],
                     "source": segment["source"], "tts_text": segment["text"],
                     "audio_start": round(begin, 6), "audio_end": round(end, 6)})
        cursor = end
    for segment in source["segments"]:
        segment["tts_text"] = spoken[segment["id"]]
    # Подготовка речи на машине ставит подпись, по которой сборка понимает, что текст не меняли:
    # считаем её тем же кодом конвейера, что и на машине, — иначе проверка бессмысленна.
    import prepare_speech as ps
    import russian_text as ru
    for segment in source["segments"]:
        segment["tts_normalized"] = ru.normalize(segment["text"], strict=False)
        segment["tts_sha256"] = ps.digest([ps.VERSION, segment["text"], segment["tts_text"]])
    source["speech_preparation"] = {"version": ps.VERSION, "mode": "offline",
                                    "input_sha256": ps.input_identity(source), "segments": len(source["segments"])}
    (part / "project" / "source" / "manifest.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    (audio / "timeline.json").write_text(json.dumps({"segments": rows}, ensure_ascii=False), encoding="utf-8")
    (audio / "benchmark.json").write_text(json.dumps(
        {"state": "complete", "final_audio_seconds": cursor,
         "book": {"input_sha256": source["book"]["input_sha256"]}}, ensure_ascii=False), encoding="utf-8")
    # Сырьё кодируется во FLAC и уезжает без sample.wav: дорожку книга собирает сама, в образе.
    for wav in (audio / "segments").glob("*.wav"):
        flac = wav.with_suffix(".flac")
        subprocess.run([a.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                        "-c:a", "flac", str(flac)], check=True)
        wav.unlink()
    (audio / "sample.wav").unlink(missing_ok=True)
    return cursor


def build_part(root, index, total, title) -> float:
    """Одна часть: epub части → проектный манифест → prepare → синтез. Как на машине."""
    part = root / f"part-{index}"
    (part / "project").mkdir(parents=True, exist_ok=True)
    epub_path = make_epub(part / f"часть-{index}.epub", title, [CHAPTERS[index - 1]])
    from build_single_voice_manifest import build
    build(str(epub_path), part / "project")
    a.prepare(part / "project" / "manifest.json", part / "project" / "source",
              highlight="sentence", start="p00001")
    return fake_synthesis(part)


@contextmanager
def parts_book():
    """Две части книги и сама книга — в отдельном каталоге, как они лежат в томе бота."""
    root = WORK / "parts" / "kniga"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    book = make_epub(WORK / "kniga.epub", "Книга целиком", CHAPTERS)
    seconds = [build_part(root, index, len(CHAPTERS), f"Книга целиком — часть {index} из {len(CHAPTERS)}")
               for index in (1, 2)]
    yield root, book, seconds


def glue(*arguments) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "glue_book.py"), *[str(item) for item in arguments]],
                          capture_output=True, text=True, cwd=str(REPO))


class GlueTests(unittest.TestCase):
    @unittest.skipUnless(pathlib.Path(a.FFMPEG).exists(), "нужен ffmpeg")
    def test_two_parts_become_one_book(self):
        with parts_book() as (root, book, seconds):
            out = WORK / "results"
            done = glue("--mode", "final", "--parts-dir", root, "--book", book, "--out", out,
                        "--work", WORK / "glue", "--expect", 2)
            self.assertEqual(done.returncode, 0, done.stdout[-4000:] + done.stderr[-2000:])
            archive = out / "Книга целиком.zip"
            self.assertTrue(archive.is_file(), list(out.iterdir()))
            with zipfile.ZipFile(archive) as zf:
                names = sorted(zf.namelist())
            self.assertEqual(names, ["Книга целиком-read-along.epub", "Книга целиком.m4b"])
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(WORK / "extracted")
            total = sum(seconds)
            probe = json.loads(subprocess.check_output(
                [a.FFPROBE, "-v", "error", "-show_format", "-show_chapters", "-of", "json",
                 str(WORK / "extracted" / "Книга целиком.m4b")]))
            # Длительность книги — сумма частей, главы — главы обеих частей.
            self.assertAlmostEqual(float(probe["format"]["duration"]), total, delta=0.3)
            self.assertEqual(len(probe["chapters"]), len(CHAPTERS))
            # Подсветка: в read-along есть текст обеих частей, и каждая строка привязана к аудио.
            with zipfile.ZipFile(WORK / "extracted" / "Книга целиком-read-along.epub") as zf:
                text = "".join(zf.read(name).decode("utf-8", "replace") for name in zf.namelist()
                               if name.endswith((".xhtml", ".html")))
                overlays = [name for name in zf.namelist() if name.endswith(".smil")]
            self.assertIn("Арина вошла", text)
            self.assertIn("Она села у окна", text)
            self.assertEqual(len(overlays), len(CHAPTERS), overlays)
            glued = json.loads((root / "glue.json").read_text(encoding="utf-8"))
            counted = sum(len(json.loads((root / f"part-{index}" / "project" / "source" / "manifest.json")
                                        .read_text(encoding="utf-8"))["segments"])
                          for index in (1, 2))
            self.assertEqual((glued["parts"], glued["segments"], glued["polish"]), (2, counted, "soft"))

    @unittest.skipUnless(pathlib.Path(a.FFMPEG).exists(), "нужен ffmpeg")
    def test_part_is_a_book_of_its_own(self):
        with parts_book() as (root, book, seconds):
            out = WORK / "results-part"
            done = glue("--mode", "part", "--part", 2, "--parts-dir", root, "--out", out,
                        "--work", WORK / "glue-part")
            self.assertEqual(done.returncode, 0, done.stdout[-4000:] + done.stderr[-2000:])
            archive = out / "Книга целиком — часть 2 из 2.zip"
            self.assertTrue(archive.is_file(), list(out.iterdir()))
            with zipfile.ZipFile(archive) as zf:
                self.assertIn("Книга целиком — часть 2 из 2-read-along.epub", zf.namelist())
            # Часть отдаётся как есть: громкость и музыка не трогались.
            self.assertFalse((WORK / "glue-part" / "audio-normalized").exists())

    def test_seam_mismatch_stops_the_assembly(self):
        with parts_book() as (root, book, _seconds):
            prepared = root / "part-2" / "project" / "source" / "manifest.json"
            payload = json.loads(prepared.read_text(encoding="utf-8"))
            payload["segments"][0]["text"] = "Совсем другой текст."
            prepared.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            done = glue("--mode", "final", "--parts-dir", root, "--book", book,
                        "--out", WORK / "results-bad", "--work", WORK / "glue-bad")
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("стык не сходится", done.stdout + done.stderr)
            self.assertIn("собирать книгу с разъехавшейся подсветкой нельзя".lower(),
                          (done.stdout + done.stderr).lower())

    def test_missing_part_is_not_a_book(self):
        with parts_book() as (root, book, _seconds):
            shutil.rmtree(root / "part-2")
            done = glue("--mode", "final", "--parts-dir", root, "--book", book,
                        "--out", WORK / "results-short", "--expect", 2)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("книга резалась на 2", done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
