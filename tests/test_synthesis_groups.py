"""Нарезка до модели: короткие предложения и реплики уходят модели одной склейкой.

Модель (Qwen3-TTS) рвётся на одно-двухсловных кусках, поэтому подготовка склеивает их до
~220 знаков, а подсветка остаётся по предложениям. Здесь проверяется вся цепочка на
поддельной модели: prepare → группы синтеза → один вызов на группу → деление звука по
паузам → разметка времени → упаковка с обложкой.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
import zipfile

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "colab"))

import audiobook_epub as a  # noqa: E402
import synthesize_qwen_mlx as q  # noqa: E402

TEXTS = [
    "Он вошёл. Было темно. Пахло сыростью и старой бумагой. Где-то за стеной капала вода. "
    "Он постоял у двери, прислушался и только потом зажёг свечу. Комната оказалась пустой. "
    "На столе лежал развёрнутый журнал, рядом очки и остывший чай. За окном кричали воробьи, "
    "и этот простой звук неожиданно успокоил его. Он сел. Подумал. Встал снова.",
    "— Да.",
    "— Нет, — сказал он.",
    "— Почему?",
    "Она не ответила. Только пожала плечами и отвернулась к окну, за которым медленно падал снег.",
]
SR = 24000
# Минимальный валидный JPEG: SOI + APP0 JFIF + EOI достаточно для ffmpeg -c:v copy не годится,
# поэтому обложку делает сам ffmpeg (см. make_cover).


def make_cover(path):
    import subprocess
    subprocess.run([a.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=navy:s=64x96", "-frames:v", "1", str(path)], check=True)
    return path.read_bytes()


def make_epub(path, cover):
    """EPUB 2 с обложкой через <meta name="cover">, как у fb2epub-конвертеров."""
    chapter = "".join(f'<p id="b{i}">{t}</p>' for i, t in enumerate(TEXTS))
    opf = ('<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Тест</dc:title>'
           '<dc:creator>Автор</dc:creator><meta name="cover" content="img1"/></metadata>'
           '<manifest><item id="img1" href="images/cover.jpg" media-type="image/jpeg"/>'
           '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>'
           '<spine><itemref idref="c1"/></spine></package>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
                   '<rootfiles><rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OPS/content.opf", opf)
        z.writestr("OPS/images/cover.jpg", cover)
        z.writestr("OPS/ch1.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Глава</h1>'
                   + chapter + "</body></html>")


class FakeResult:
    def __init__(self, audio):
        self.audio, self.sample_rate, self.token_count = audio, SR, 10


class FakeModel:
    """Каждое предложение — тон длиной по числу букв, после него 0,35 с тишины."""

    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(tts_model_type="base")
        self.speech_tokenizer = SimpleNamespace(has_encoder=True)
        self.tokenizer = object()

    def generate(self, text, **options):
        self.calls.append(text)
        pieces = []
        for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
            letters = sum(c.isalnum() for c in sentence)
            if not letters:
                continue
            t = np.arange(int(SR * 0.06 * letters)) / SR
            pieces += [0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32), np.zeros(int(SR * 0.35), np.float32)]
        yield FakeResult(np.concatenate(pieces))


class DriftingModel(FakeModel):
    """Модель, у которой голос в первой склейке «уезжает»: громче обычного — значит, чужой тембр.

    Отпечаток голоса подделан по громкости: образец и нормальная речь — [1, 0], уехавший голос —
    [0, 1] (сходство 0). Так видно, ЧТО страховка выбрала, а не только что она сработала.
    """

    def __init__(self):
        super().__init__()
        self.speaker_encoder = object()
        self.drifted = False

    def generate(self, text, **options):
        for result in super().generate(text, **options):
            if len(self.calls) == 2 and not self.drifted:      # первая склейка после прогрева
                self.drifted = True
                result.audio = result.audio * (0.6 / 0.3)
            yield result

    def extract_speaker_embedding(self, audio, sr=24000):
        return np.array([0.0, 1.0]) if float(np.max(np.abs(audio))) > 0.45 else np.array([1.0, 0.0])


@contextmanager
def fake_mlx(model):
    core = ModuleType("mlx.core")
    core.gpu = "gpu"
    core.random = SimpleNamespace(seed=lambda value: None)
    for name in ("set_default_device", "set_cache_limit", "set_memory_limit", "synchronize", "clear_cache"):
        setattr(core, name, lambda *args, **kwargs: None)
    core.device_info = lambda: {"device_name": "fake"}
    core.get_peak_memory = core.get_active_memory = lambda: 0
    core.array = lambda value: value
    mlx = ModuleType("mlx")
    mlx.core = core
    utils = ModuleType("mlx_audio.tts.utils")
    utils.load_model = lambda path: model
    audio_utils = ModuleType("mlx_audio.utils")
    audio_utils.load_audio = lambda path, sample_rate=24000: sf.read(str(path), dtype="float32")[0]
    names = {"mlx": mlx, "mlx.core": core, "mlx_audio": ModuleType("mlx_audio"),
             "mlx_audio.tts": ModuleType("mlx_audio.tts"), "mlx_audio.tts.utils": utils,
             "mlx_audio.utils": audio_utils}
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules.update(names)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@unittest.skipIf(np is None, "numpy/soundfile нужны для синтеза")
class SplitAudioTests(unittest.TestCase):
    def test_cut_lands_in_the_pause_not_where_text_share_predicts(self):
        tone = lambda s: np.full(int(SR * s), 0.3, np.float32)
        silence = lambda s: np.zeros(int(SR * s), np.float32)
        audio = np.concatenate([tone(1.0), silence(0.4), tone(2.0), silence(0.4), tone(0.5)])
        # Текст «обещает» доли 1:1:1, звук — 1:2:0,5: резать надо по паузам, а не по долям.
        parts = q.split_group_audio(audio, SR, [10, 10, 10])
        self.assertEqual(sum(len(p) for p in parts), len(audio))
        self.assertTrue(np.array_equal(np.concatenate(parts), audio))
        first, second = len(parts[0]) / SR, (len(parts[0]) + len(parts[1])) / SR
        self.assertTrue(1.0 <= first <= 1.4, first)
        self.assertTrue(3.4 <= second <= 3.8, second)

    def test_every_piece_has_sound_even_for_a_tiny_line(self):
        audio = np.concatenate([np.full(SR, 0.3, np.float32), np.zeros(SR // 10, np.float32),
                                np.full(SR // 5, 0.3, np.float32)])
        parts = q.split_group_audio(audio, SR, [40, 1])
        self.assertTrue(all(len(p) > 0 for p in parts))
        self.assertEqual(sum(len(p) for p in parts), len(audio))


class JoinTests(unittest.TestCase):
    def row(self, sid, speaker, text, chapter="ch1"):
        return {"id": sid, "speaker": speaker, "text": text, "chapter_id": chapter, "pause_after_ms": 300,
                "synthesis_group": sid}

    def test_same_reference_joins_dialogue_and_narration_but_not_other_voices(self):
        same = {"wav_sha256": "x", "ref_text": "образец"}
        refs = {"narrator": same, "dialog": dict(same), "arina": {"wav_sha256": "y", "ref_text": "другой"}}
        rows = [self.row("a", "dialog", "— Да,"), self.row("b", "narrator", "— сказал он."),
                self.row("c", "arina", "— Нет."), self.row("d", "narrator", "Тишина.", chapter="ch2")]
        groups = q.join_same_reference(q.synthesis_groups(rows), refs, 100, 260)
        self.assertEqual([[s["id"] for _, s in g] for g in groups], [["a", "b"], ["c"], ["d"]])
        self.assertEqual(rows[0]["pause_after_ms"], 0)      # стык внутри одного вызова модели
        self.assertEqual(rows[1]["pause_after_ms"], 300)

    def test_long_neighbours_stay_separate(self):
        refs = {"narrator": {"wav_sha256": "x", "ref_text": "t"}}
        rows = [self.row("a", "narrator", "а" * 150), self.row("b", "narrator", "б" * 150)]
        self.assertEqual(len(q.join_same_reference(q.synthesis_groups(rows), refs, 100, 260)), 2)


@unittest.skipIf(np is None, "numpy/soundfile нужны для синтеза")
class PipelineTests(unittest.TestCase):
    def build(self, folder, highlight):
        import build_single_voice_manifest as b
        folder = Path(folder)
        cover = make_cover(folder / "cover-src.jpg")
        make_epub(folder / "book.epub", cover)
        b.build(folder / "book.epub", folder / "project")
        project = json.loads((folder / "project/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(project["book"]["cover"]["sha256"], hashlib.sha256(cover).hexdigest())
        a.prepare(str(folder / "project/manifest.json"), str(folder / "source"), highlight=highlight, start="p00001")
        manifest = json.loads((folder / "source/manifest.json").read_text(encoding="utf-8"))
        library = folder / "library"
        library.mkdir()
        tone = (0.2 * np.sin(np.arange(SR * 3) / SR * 2 * np.pi * 180)).astype(np.float32)
        sf.write(library / "narrator.wav", tone, SR)
        for voice in manifest["voices"]:
            target = library / (voice["id"] + ".wav")
            if voice["id"] != "narrator":
                shutil.copyfile(library / "narrator.wav", target)
            q.save(library / (voice["id"] + ".json"), {
                "profile": q.library_profile(voice, manifest["book"]["input_sha256"]), "ref_text": "Образец.",
                "wav_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        return folder, manifest, cover

    def render(self, folder, model):
        args = SimpleNamespace(manifest=str(folder / "source/manifest.json"), model=str(folder), out=str(folder / "audio"),
                               seconds=300, whole_book=True, start_paragraph="p00001", temperature=0.9,
                               max_tokens=1024, stress_marks="keep", plan=False, mode="clone",
                               rab_generation=False, voice_library=str(folder / "library"), design_model=None,
                               reference_source=None, import_only=False, ephemeral_voices=False)
        with fake_mlx(model):
            q.render(args)

    def check(self, highlight):
        with tempfile.TemporaryDirectory() as folder:
            folder, manifest, cover = self.build(folder, highlight)
            model = FakeModel()
            self.render(folder, model)
            calls = model.calls[1:]                                   # первый — прогрев
            self.assertTrue(all(len(c) >= 50 for c in calls), calls)  # ни одного «— Да.» в одиночку
            self.assertLess(len(calls), len(manifest["segments"]))
            words = lambda text: re.findall(r"\w+", text)
            self.assertEqual(words(" ".join(calls)), words(" ".join(s["text"] for s in manifest["segments"])))
            # Заголовок без точки, склеенный с абзацем, получает точку только в тексте для модели.
            self.assertTrue(calls[0].startswith("Глава. Он вошёл."), calls[0])
            timeline = json.loads((folder / "audio/timeline.json").read_text())["segments"]
            self.assertEqual([r["segment_id"] for r in timeline], [s["id"] for s in manifest["segments"]])
            for row in timeline:
                frames = sf.info(folder / "audio/segments" / (row["segment_id"] + ".wav")).frames
                self.assertAlmostEqual(frames / SR, row["audio_end"] - row["audio_start"], places=6)
            # Повтор целиком из кэша: модель зовётся только на прогрев.
            again = FakeModel()
            self.render(folder, again)
            self.assertEqual(len(again.calls), 1)
            a.package(str(folder / "source/manifest.json"), str(folder / "audio"), str(folder / "result"))
            import subprocess
            probe = json.loads(subprocess.check_output([a.FFPROBE, "-v", "error", "-show_streams", "-of", "json",
                                                        str(folder / "result/book.m4b")]))
            self.assertTrue(any(s.get("disposition", {}).get("attached_pic") for s in probe["streams"]))
            with zipfile.ZipFile(folder / "result/book-read-along.epub") as z:
                opf = next(n for n in z.namelist() if n.endswith(".opf"))
                text = z.read(opf).decode("utf-8")
                self.assertRegex(text, r'properties="cover-image"|name="cover"')
                image = next(n for n in z.namelist() if n.endswith("cover.jpg"))
                self.assertEqual(z.read(image), cover)

    def test_drifted_voice_is_regenerated_and_the_matching_take_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            folder, manifest, _ = self.build(folder, "sentence")
            model = DriftingModel()
            self.render(folder, model)
            calls = model.calls[1:]                               # первый — прогрев
            self.assertEqual(calls[0], calls[1], "уехавшую склейку синтезировали заново тем же текстом")
            report = json.loads((folder / "audio/benchmark.json").read_text())
            check = report["voice_check"]
            self.assertTrue(check["enabled"])
            self.assertGreater(check["regenerated_rows"], 0)
            self.assertGreaterEqual(check["min"], 0.99)           # в книгу попал совпавший с образцом дубль
            first = json.loads((folder / "audio/segments" / (manifest["segments"][0]["id"] + ".json")).read_text())
            self.assertEqual(first["metrics"]["voice_attempts"], 2)
            peak = float(np.max(np.abs(sf.read(folder / "audio/segments" / (manifest["segments"][1]["id"] + ".wav"))[0])))
            self.assertLess(peak, 0.45)

    def test_sentence_highlight_sends_glued_text_to_model(self):
        self.check("sentence")

    def test_fragment_highlight_with_glued_paragraphs(self):
        self.check("fragment")


if __name__ == "__main__":
    unittest.main()
