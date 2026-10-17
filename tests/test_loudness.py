import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from normalize_book_audio import gains_for, polish


def tone_wav(path, seconds=3.0, rate=24000):
    """Рече-подобный сигнал: низкая «речь» 200 Гц + пищалка 9.5 кГц (свистящий призвук)."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        data = bytearray()
        for index in range(frames):
            moment = index / rate
            value = 0.25 * math.sin(2 * math.pi * 200 * moment) + 0.12 * math.sin(2 * math.pi * 9500 * moment)
            data += struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
        handle.writeframes(bytes(data))
    return path, frames, rate


class LoudnessTests(unittest.TestCase):
    def test_target_is_common_and_preserves_peak_headroom(self):
        measurements = {"a": {"lufs": -28, "true_peak_db": -8}, "b": {"lufs": -18, "true_peak_db": -4}}
        target, gains = gains_for(measurements)
        for key, value in measurements.items():
            self.assertAlmostEqual(value["lufs"] + gains[key], target)
            self.assertLessEqual(value["true_peak_db"] + gains[key], -1.5)

    def test_silent_role_is_not_amplified(self):
        _, gains = gains_for({"speech": {"lufs": -20, "true_peak_db": -4},
                              "silence": {"lufs": float('-inf'), "true_peak_db": float('-inf')}})
        self.assertEqual(gains['silence'], 0)
        with self.assertRaises(ValueError):
            gains_for({"silent": {"lufs": float('-inf'), "true_peak_db": float('-inf')}})


def band_level(x, rate, low, high):
    """Уровень полосы: у тестового сигнала 200 Гц — «речь», 9.5 кГц — тот самый свист."""
    import numpy as np
    spectrum = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), 1 / rate)
    spectrum[(freq < low) | (freq >= high)] = 0
    return 20 * np.log10(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(x)) ** 2)) + 1e-9)


class PolishTests(unittest.TestCase):
    """Полировка — последняя обработка готовой книги: она не смеет менять длительность."""

    def test_high_frequency_squeal_is_softened_without_touching_timing(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path, frames, rate = tone_wav(Path(folder) / "montage.wav")
            original = np.frombuffer(wave.open(str(path)).readframes(frames), dtype="<i2").astype("float32")
            report = polish(path, target=-20.0, peak_limit=-1.5)
            self.assertTrue(report["timing_unchanged"])
            reader = wave.open(str(path))
            self.assertEqual(reader.getnframes(), frames)
            self.assertEqual(reader.getframerate(), rate)
            polished = np.frombuffer(reader.readframes(frames), dtype="<i2").astype("float32")
            before_squeal = band_level(original, rate, 8500, 11000)
            after_squeal = band_level(polished, rate, 8500, 11000)
            before_speech = band_level(original, rate, 150, 400)
            after_speech = band_level(polished, rate, 150, 400)
            self.assertLess(after_squeal - before_squeal, -2.0, "полка не сняла свист")
            # Свист должен пострадать заметно сильнее голоса: иначе это не полировка, а глушение.
            self.assertGreater((after_speech - before_speech) - (after_squeal - before_squeal), 5.0,
                               "вместе со свистом уехал и голос")
            # Громкость после фильтра возвращается к цели постоянным усилением, а не «как получилось».
            self.assertAlmostEqual(report["lufs_after"], -20.0, delta=1.0)
            self.assertLessEqual(report["true_peak_db"], -1.4)

    def test_short_file_is_not_broken(self):
        with tempfile.TemporaryDirectory() as folder:
            path, frames, _ = tone_wav(Path(folder) / "tiny.wav", seconds=0.5)
            polish(path, target=-20.0, peak_limit=-1.5)
            self.assertEqual(wave.open(str(path)).getnframes(), frames)
