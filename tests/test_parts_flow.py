"""Режим частей настоящим путём бота: shrink_book → машина (подготовка, синтез, pack_raw) → glue.

В test_glue_book части собираются фикстурой, мимо нарезки и упаковки сырья. Здесь — те же шаги,
что у бота: книга с обложкой режется shrink_book, каждая часть готовится и синтезируется (модель
поддельная, см. test_synthesis_groups), сырьё пакуется настоящим pack_raw, распаковывается в том
и собирается glue_book — сначала каждая часть, потом вся книга.
"""
import collections
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "colab"), str(REPO / "colab" / "cli"), str(REPO / "tests")]

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None

import audiobook_epub as a  # noqa: E402


def run(*command):
    done = subprocess.run([str(c) for c in command], cwd=str(REPO), capture_output=True, text=True)
    if done.returncode:
        raise AssertionError(f"{command[1]} упал:\n{(done.stdout + done.stderr)[-2000:]}")
    return done


def probe(path):
    return json.loads(subprocess.check_output([a.FFPROBE, "-v", "error", "-show_format", "-show_streams",
                                               "-show_chapters", "-of", "json", str(path)]))


@unittest.skipIf(np is None or not shutil.which(a.FFMPEG), "нужны numpy, soundfile и ffmpeg")
class PartsFlowTests(unittest.TestCase):
    def test_book_cut_by_shrink_goes_through_parts_and_glue(self):
        import build_single_voice_manifest as b
        import qwen_tts_book
        import synthesize_qwen_mlx as q
        import test_synthesis_groups as t

        root = pathlib.Path(tempfile.mkdtemp(prefix="parts-flow-"))
        cover = t.make_cover(root / "cover-src.jpg")
        t.make_epub(root / "kniga.epub", cover)
        run(sys.executable, REPO / "shrink_book.py", root / "kniga.epub", "--out", root / "books",
            "--parts", 2, "--no-drive")
        part_books = sorted((root / "books").glob("*.epub"))
        self.assertEqual(len(part_books), 2)
        for book in part_books:
            with zipfile.ZipFile(book) as zf:
                twice = [n for n, c in collections.Counter(i.filename for i in zf.infolist()).items() if c > 1]
                self.assertEqual(twice, [], f"{book.name}: файлы дважды")

        parts_dir, seconds = root / "parts", []
        for index, part_book in enumerate(part_books, 1):
            machine = root / f"machine-{index}" / "out"
            project = machine / "project"
            b.build(part_book, project)
            a.prepare(str(project / "manifest.json"), str(project / "source"), highlight="sentence", start="p00001")
            manifest = json.loads((project / "source/manifest.json").read_text(encoding="utf-8"))
            library = project / "voice-library"
            library.mkdir()
            sf.write(library / "narrator.wav", (0.2 * np.sin(np.arange(72000) / 24000 * 6.28 * 180)).astype(np.float32), 24000)
            for voice in manifest["voices"]:
                target = library / (voice["id"] + ".wav")
                if voice["id"] != "narrator":
                    shutil.copyfile(library / "narrator.wav", target)
                q.save(library / (voice["id"] + ".json"), {
                    "profile": q.library_profile(voice, manifest["book"]["input_sha256"]), "ref_text": "Образец.",
                    "wav_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            args = SimpleNamespace(manifest=str(project / "source/manifest.json"), model=str(machine),
                                   out=str(project / "audio"), seconds=300, whole_book=True, start_paragraph="p00001",
                                   temperature=0.9, max_tokens=1024, stress_marks="keep", plan=False, mode="clone",
                                   rab_generation=False, voice_library=str(library), design_model=None,
                                   reference_source=None, import_only=False, ephemeral_voices=False)
            with t.fake_mlx(t.FakeModel()):
                q.render(args)
            raw = machine.parent / "raw.zip"
            qwen_tts_book.pack_raw(SimpleNamespace(out=str(machine), zip=str(raw)), part_book)
            (parts_dir / f"part-{index}").mkdir(parents=True)
            with zipfile.ZipFile(raw) as zf:
                zf.extractall(parts_dir / f"part-{index}")
            run(sys.executable, REPO / "glue_book.py", "--mode", "part", "--part", index, "--parts-dir", parts_dir,
                "--out", root / "results", "--name", f"part{index}")
            seconds.append(json.loads((project / "audio/benchmark.json").read_text())["final_audio_seconds"])

        run(sys.executable, REPO / "glue_book.py", "--mode", "final", "--parts-dir", parts_dir,
            "--book", root / "kniga.epub", "--out", root / "results", "--expect", 2, "--name", "kniga")

        for name in ("part1", "part2", "kniga"):
            with zipfile.ZipFile(root / "results" / f"{name}.zip") as zf:
                m4b = next(n for n in zf.namelist() if n.endswith(".m4b"))
                epub_name = next(n for n in zf.namelist() if n.endswith(".epub"))
                zf.extract(m4b, root / name)
                with zf.open(epub_name) as handle, zipfile.ZipFile(handle) as inner:
                    opf = next(n for n in inner.namelist() if n.endswith(".opf"))
                    self.assertIn('properties="cover-image"', inner.read(opf).decode(), f"{name}: нет обложки в epub")
                    image = next(n for n in inner.namelist() if "cover" in n and n.endswith(".jpg"))
                    self.assertEqual(inner.read(image), cover, f"{name}: не та обложка")
            info = probe(root / name / m4b)
            self.assertTrue(any(s.get("disposition", {}).get("attached_pic") for s in info["streams"]),
                            f"{name}: нет обложки в m4b")
            if name == "kniga":
                self.assertAlmostEqual(float(info["format"]["duration"]), sum(seconds), delta=0.2)
                # В книге одна глава; стык разрезал её между частями — в итоге она снова одна.
                self.assertEqual(len(info["chapters"]), 1)


if __name__ == "__main__":
    unittest.main()
