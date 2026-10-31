"""Стерео-объём: моно-книга расширяется, но длительность, громкость и пик не меняются.

    python3 tests/test_stereo.py
"""
import json
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "colab"))

import audio_limits                                      # noqa: E402
import widen_book_audio                                   # noqa: E402

RATE = 24000


def mono_book(folder, seconds=2.0, lufs=-20.0):
    """Готовая моно-дорожка с музыкой: речь 200 Гц + «шум» 3 кГц, как после микса."""
    import numpy as np
    import soundfile as sf

    frames = int(seconds * RATE)
    moment = np.arange(frames) / RATE
    speech = 0.2 * np.sin(2 * np.pi * 200 * moment) + 0.06 * np.sin(2 * np.pi * 3000 * moment)
    sf.write(str(folder / "sample.wav"), speech, RATE, subtype="PCM_16")
    (folder / "timeline.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
    (folder / "benchmark.json").write_text(json.dumps(
        {"state": "complete", "final_audio_seconds": frames / RATE}), encoding="utf-8")
    (folder / "loudness.json").write_text(json.dumps({"actual_lufs": lufs}), encoding="utf-8")
    return frames


class StereoTests(unittest.TestCase):
    def test_book_becomes_stereo_without_changing_length_or_loudness(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            source = folder / "music"
            source.mkdir()
            frames = mono_book(source)
            report = widen_book_audio.widen(source, folder / "stereo")
            audio, rate = sf.read(str(folder / "stereo" / "sample.wav"))
            self.assertEqual(audio.shape, (frames, 2))
            self.assertEqual(rate, RATE)
            self.assertTrue(report["timing_unchanged"])
            # Громкость осталась той же, что у моно-книги: два канала сами дают +3…4 дБ.
            self.assertAlmostEqual(report["lufs_after"], -20.0, delta=0.7)
            self.assertLessEqual(report["true_peak_db"], -1.4)
            self.assertLess(report["peak"], 0.98)
            # Каналы разные (это не дубль моно), но корреляция высокая: объём, а не «развал».
            correlation = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
            self.assertLess(correlation, 0.995)
            self.assertGreater(correlation, 0.75)
            self.assertTrue((folder / "stereo" / "timeline.json").is_file())
            self.assertEqual(json.loads((folder / "stereo" / "benchmark.json").read_text())["stereo"]["channels"], 2)

    def test_stereo_input_is_refused(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            source = folder / "music"
            source.mkdir()
            mono_book(source)
            wide = folder / "wide"
            widen_book_audio.widen(source, wide)
            with self.assertRaises(ValueError):
                widen_book_audio.widen(wide, folder / "wide2")      # стерео уже сделано

    def test_long_stereo_track_goes_to_rf64(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            source = folder / "music"
            source.mkdir()
            mono_book(source, seconds=1.0)
            limit = audio_limits.LIMIT
            audio_limits.LIMIT = 20_000                     # 1 с стерео = 96 000 байт: уже RF64
            try:
                widen_book_audio.widen(source, folder / "stereo")
            finally:
                audio_limits.LIMIT = limit
            path = folder / "stereo" / "sample.wav"
            self.assertEqual(path.read_bytes()[:4], b"RF64")
            self.assertEqual(sf.info(str(path)).channels, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
