"""Предел обычного WAV: 4 ГиБ.

Дорожка «Идиота» — 25.61 ч при 24 кГц/16 бит, это 4 425 413 424 байта. В RIFF заголовок влезает
только 0xFFFFFFFF, и сборка падала «дорожка книги вышла 89 478.485 с вместо 92 196.113 с» —
45 минут книги пропадали из вида, хотя звук был записан. Теперь такие дорожки пишутся как
RF64. Здесь проверяется сама ветка RF64: порог опущен до килобайт, чтобы не писать 4 ГБ.

    python3 tests/test_wav_limit.py
"""
import json
import pathlib
import sys
import tempfile
import unittest
import wave

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "colab"))

import audio_limits                                      # noqa: E402

RATE = 24000


def make_audio(folder, seconds_list):
    """Сырьё части: сегменты + timeline, моно 24 кГц PCM_16, без пауз между фразами."""
    import numpy as np
    import soundfile as sf

    (folder / "segments").mkdir(parents=True)
    rows, cursor = [], 0.0
    for index, seconds in enumerate(seconds_list, 1):
        name = f"seg{index:06d}"
        frames = round(seconds * RATE)
        tone = 0.05 * np.sin(np.arange(frames) / 20.0)
        sf.write(str(folder / "segments" / (name + ".wav")), tone, RATE, subtype="PCM_16")
        begin, end = round(cursor * RATE) / RATE, round((cursor + seconds) * RATE) / RATE
        rows.append({"segment_id": name, "speaker": "narrator",
                     "audio_start": begin, "audio_end": end})
        cursor = end
    (folder / "timeline.json").write_text(json.dumps({"segments": rows}), encoding="utf-8")
    return folder, cursor


class WavLimitTests(unittest.TestCase):
    def test_book_longer_than_24_hours_needs_rf64(self):
        # «Идиот» целиком: 92 196.113 с — ровно тот случай, на котором сборка встала.
        self.assertEqual(audio_limits.wav_format(round(92196.113 * RATE), 2), "RF64")
        # А часть в полчаса и книга до потолка остаются обычным WAV.
        self.assertEqual(audio_limits.wav_format(round(1800 * RATE), 2), "WAV")
        self.assertEqual(audio_limits.wav_format(round(89000 * RATE), 2), "WAV")
        self.assertFalse(audio_limits.too_big(round(89000 * RATE), 2))
        self.assertTrue(audio_limits.too_big(round(92196 * RATE), 2))

    def test_long_montage_is_written_as_rf64_and_keeps_its_length(self):
        import glue_book
        import soundfile as sf

        with tempfile.TemporaryDirectory() as folder:
            audio, total = make_audio(pathlib.Path(folder), [0.4, 0.4, 0.4])
            destination = pathlib.Path(folder) / "sample.wav"
            limit = audio_limits.LIMIT
            audio_limits.LIMIT = 20_000                     # 0.4 с = 19 200 байт: дальше уже RF64
            try:
                glue_book.build_sample(audio, destination, total)
            finally:
                audio_limits.LIMIT = limit
            self.assertEqual(destination.read_bytes()[:4], b"RF64")
            # Длительность верна вплоть до кадра — это и проверяет сам build_sample.
            self.assertEqual(sf.info(str(destination)).frames, round(total * RATE))
            # Именно поэтому длину проверять через `wave` нельзя: RF64 он не читает.
            with self.assertRaises(wave.Error):
                wave.open(str(destination))

    def test_normalization_reads_shape_from_rf64(self):
        import soundfile as sf

        from normalize_book_audio import wav_shape

        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "montage.wav"
            sf.write(str(path), [0.1] * 4800, RATE, subtype="PCM_16", format="RF64")
            self.assertEqual(wav_shape(path), (4800, RATE, 1))

    def test_music_mix_writes_rf64_for_long_book(self):
        # Порог опущен так же: ветка та же, что и на 25-часовой книге.
        import soundfile as sf

        from mix_book_music import mix

        audio, total = None, None
        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            source, total = make_audio(folder / "speech", [0.5, 0.5])
            import glue_book
            glue_book.build_sample(source, source / "sample.wav", total)
            (source / "benchmark.json").write_text(json.dumps(
                {"state": "complete", "final_audio_seconds": total}), encoding="utf-8")
            music = folder / "music.wav"
            sf.write(str(music), [0.02] * int(RATE * 1.0), RATE, subtype="PCM_16")
            limit = audio_limits.LIMIT
            audio_limits.LIMIT = 20_000
            try:
                mix(source, folder / "mixed", music, percent=5)
            finally:
                audio_limits.LIMIT = limit
            mixed = folder / "mixed" / "sample.wav"
            self.assertEqual(mixed.read_bytes()[:4], b"RF64")
            self.assertEqual(sf.info(str(mixed)).frames, round(total * RATE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
