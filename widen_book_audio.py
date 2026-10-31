#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""widen_book_audio.py — стерео-объём для готовой книги: моно → два канала с шириной.

Источник — моно 24 кГц от синтеза, поэтому настоящее стерео взять неоткуда: расширение
делается эффектом Хааса (правая сторона на 12 мс позже, боковая составляющая тише) и НЕ
добавляет информации. Зато книга перестаёт звучать «в одну точку»: у голоса появляется
объём, у музыки — воздух.

Что важно и проверяется:
  - длительность НЕ меняется ни на кадр: от неё зависит подсветка read-along;
  - громкость остаётся той же, что у моно-дорожки: два канала дают +3…4 дБ сами по себе,
    поэтому после расширения ставим постоянное усиление к цели (не лимитер);
  - пик не выходит за -1.5 дБ, сумма каналов не перегружена;
  - выход — RF64, если дорожка больше 4 ГиБ (25 ч книги), см. audio_limits.

    python3 widen_book_audio.py --audio-dir <моно с музыкой> --out <стерео>
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import audio_limits

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# Эффект Хааса: правый канал позже левого на 12 мс, боковая составляющая 0.3 — мягкая ширина
# без «радиоспектакля». 12 мс слышатся как объём, а не как эхо; больше 20 мс уже отставание.
SPACE_FILTER = ("pan=stereo|c0=c0|c1=c0,"
                "haas=side_gain=0.3:middle_source=mid:left_delay=0:right_delay=12:right_phase=false")
TARGET_LUFS = -20.0
PEAK_LIMIT = -1.5


def measure(path):
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
        "-af", "loudnorm=I=-20:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        check=True, capture_output=True, text=True)
    values = json.loads(re.findall(r'\{\s*"input_i"[\s\S]*?\}', result.stderr)[-1])
    return {"lufs": float(values["input_i"]), "true_peak_db": float(values["input_tp"])}


def peak(path):
    import numpy as np
    import soundfile as sf

    highest = 0.0
    with sf.SoundFile(str(path)) as reader:
        for block in reader.blocks(blocksize=65536, dtype="float32"):
            if not np.isfinite(block).all():
                raise ValueError("Некорректные значения аудио после расширения")
            highest = max(highest, float(np.max(np.abs(block), initial=0)))
    return highest


def widen(source, out, target=TARGET_LUFS, peak_limit=PEAK_LIMIT, chain=SPACE_FILTER):
    import soundfile as sf

    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("Сохраните дорожку без стерео: выход должен быть в другом каталоге")
    benchmark = json.loads((source / "benchmark.json").read_text(encoding="utf-8"))
    if benchmark.get("state") != "complete":
        raise ValueError("Сборка речи не завершена")
    if benchmark.get("stereo"):
        raise ValueError("Стерео уже сделано; берите исходную моно-дорожку")
    info = sf.info(str(source / "sample.wav"))
    if info.channels != 1:
        raise ValueError("Расширять можно только моно-дорожку")
    out.mkdir(parents=True, exist_ok=True)
    loudness = {}
    try:
        loudness = json.loads((source / "loudness.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if math.isfinite(float(loudness.get("actual_lufs", float("nan")))):
        target = float(loudness["actual_lufs"])           # та же громкость, что у моно-книги
    # Стерео PCM_16 — 4 байта на кадр: 25 ч книги это 8.8 ГБ, обычный WAV столько не держит.
    # ffmpeg сам о размере не догадывается (умолчание -rf64 меняется между версиями) — решаем здесь.
    rf64 = "always" if audio_limits.too_big(info.frames, 4) else "auto"
    filters = f"{chain},atrim=end_sample={info.frames}"
    with tempfile.TemporaryDirectory(prefix="book-stereo-") as temporary:
        widened = Path(temporary) / "widened.wav"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(source / "sample.wav"), "-af", filters,
                        "-c:a", "pcm_s16le", "-rf64", rf64, str(widened)], check=True)
        got = sf.info(str(widened))
        if (got.frames, got.samplerate) != (info.frames, info.samplerate) or got.channels != 2:
            raise ValueError(f"Расширение изменило дорожку: {got.frames} кадров {got.channels} кан. "
                             f"вместо {info.frames} и 2")
        before = measure(widened)                          # два канала дают +3…4 дБ сами по себе
        gain = min(target - before["lufs"], peak_limit - before["true_peak_db"])
        final = Path(temporary) / "final.wav"
        command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(widened)]
        if abs(gain) >= 0.05:
            command += ["-af", f"volume={gain:.2f}dB"]
        command += ["-c:a", "pcm_s16le", "-rf64", rf64, str(final)]
        subprocess.run(command, check=True)
        got = sf.info(str(final))
        if (got.frames, got.samplerate, got.channels) != (info.frames, info.samplerate, 2):
            raise ValueError("Возврат громкости изменил длительность стерео-дорожки")
        highest = peak(final)
        if highest >= 0.98:
            raise ValueError(f"Стерео-дорожка перегружена: пик {highest:.3f}")
        shutil.move(str(final), str(out / "sample.wav"))
    after = measure(out / "sample.wav")
    shutil.copyfile(source / "timeline.json", out / "timeline.json")
    report = {"mode": "space", "filter": chain, "channels": 2,
              "lufs_before_gain": before["lufs"], "lufs_after": after["lufs"],
              "true_peak_db": after["true_peak_db"], "gain_db": round(gain, 2),
              "peak": round(highest, 4), "frames": info.frames, "timing_unchanged": True,
              "source_speech": str(source)}
    (out / "stereo.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    benchmark.update(stereo=report, output_files={"wav": str(out / "sample.wav")})
    (out / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"Стерео: объём (Хаас 12 мс, бок 0.3), громкость {before['lufs']:.2f} → "
          f"{after['lufs']:.2f} LUFS (усиление {gain:+.2f} дБ), пик {after['true_peak_db']:.2f} дБ, "
          f"длительность не изменилась", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Стерео-объём: моно-книга → два канала")
    parser.add_argument("--audio-dir", required=True, help="готовая моно-дорожка с музыкой")
    parser.add_argument("--out", required=True, help="куда положить стерео (другой каталог)")
    parser.add_argument("--target", type=float, default=TARGET_LUFS, help="цель громкости, LUFS")
    parser.add_argument("--peak-limit", type=float, default=PEAK_LIMIT, help="предел пика, дБ")
    args = parser.parse_args()
    widen(args.audio_dir, args.out, args.target, args.peak_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
