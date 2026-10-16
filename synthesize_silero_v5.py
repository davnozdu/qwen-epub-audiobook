#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Silero v5 (torch) — альтернативный синтез речи для того же конвейера.

Скрипт делает ровно то, что делает synthesize_qwen_mlx.py на своём движке: читает
подготовленный манифест (source/manifest.json) и складывает в выходной каталог аудио,
которое принимают normalize_book_audio.py → mix_book_music.py → audiobook_epub.py package.

    python synthesize_silero_v5.py --manifest project/source/manifest.json --out project/audio \
        --speaker aidar

Почему отдельный скрипт, а не замена: у Silero v5 свои плюсы и минусы.

Плюсы (измерено на этом маке, CPU, 4 потока):
  * RTF ≈ 0,01–0,1 — то есть в 10–100 раз быстрее реального времени против RTF ≈ 1 у Qwen3-TTS;
    10-часовая книга — это десятки минут, а не 10 часов GPU;
  * модель 145 МБ против 4,6 ГБ, работает на CPU, GPU не обязателен;
  * голоса: aidar, baya, kseniya, eugene, xenia; частоты 8/24/48 кГц.

Минусы и ограничения:
  * качество и выразительность ниже, чем у Qwen3-TTS 1.7B — это другой класс модели;
  * на один вызов — не больше ~60 с звука, поэтому длинные сегменты режутся и склеиваются;
  * ударения понимает только в своей нотации «+» перед гласной (наш манифест хранит
    знак U+0301 после гласной — скрипт переводит его сам);
  * при put_accent=True модель расставляет ударения по-своему и игнорирует наши,
    поэтому по умолчанию put_accent=False и ударения берутся из манифеста (тот же
    словарь Supertonic + Silero Stress, что и в основном проде).

Лицензии: v5_ru.pt идёт под CC-NC-BY (без коммерческого использования), а модели
*_cis_base* — под MIT и работают с русским (там ~20 русских голосов ru_*). Если аудиокнига
пойдёт в коммерцию:

    --model-url https://models.silero.ai/models/tts/ru/v5_cis_base.pt --speaker ru_igor
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
MODEL_MIN_BYTES = 50_000_000      # v5_ru 145 МБ, v5_cis_base 92 МБ — оба проходят
SPEAKERS = ("aidar", "baya", "kseniya", "eugene", "xenia")
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
ACUTE = "\u0301"
# Предел модели: ~5000 кадров ≈ 60 с звука на вызов (проверено: 60 с проходит, 70 с — отказ).
MAX_SECONDS_PER_CALL = 45.0
CHARS_PER_SECOND = 14.0


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def to_silero_marks(text: str) -> str:
    """U+0301 после гласной → «+» перед гласной (нотация Silero). У «ё» знак просто снимаем."""
    out = []
    for char in text:
        if char == ACUTE:
            if out and out[-1].lower() in VOWELS:
                letter = out.pop()
                if letter.lower() != "ё":
                    out.append("+" + letter)
                else:
                    out.append(letter)
            continue
        out.append(char)
    return "".join(out)


def split_long(text: str, limit_seconds: float = MAX_SECONDS_PER_CALL) -> list[str]:
    """Модель не берёт больше ~60 с звука за вызов: режем по предложениям."""
    if len(text) <= limit_seconds * CHARS_PER_SECOND:
        return [text]
    pieces = re.split(r"(?<=[.!?…»])\s+", text)
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > limit_seconds * CHARS_PER_SECOND:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def load_model(path: Path, url: str, device_name: str, threads: int):
    import torch

    torch.set_num_threads(max(1, threads))
    path = path.expanduser()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"скачиваю модель Silero v5: {url} → {path}", flush=True)
        torch.hub.download_url_to_file(url, str(path))
        size = path.stat().st_size
        if size < MODEL_MIN_BYTES:
            path.unlink(missing_ok=True)
            raise SystemExit(f"скачался подозрительно маленький файл: {size} байт — проверьте {url}")
    # Существующий файл не трогаем: он может быть другой версией модели (например v5_cis_base).
    started = time.time()
    try:
        model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    except Exception as exc:
        raise SystemExit(f"не удалось прочитать модель {path}: {exc}\n"
                         f"  это должен быть файл из models.yml (v5_ru.pt, v5_cis_base.pt и т. п.), "
                         f"а не распакованный каталог")
    device = torch.device(device_name)
    model.to(device)
    print(f"модель: {path.name} ({path.stat().st_size / 1e6:.0f} МБ) загружена за "
          f"{time.time() - started:.2f} с | устройство: {device} "
          f"| голоса: {', '.join(model.speakers)}")
    return model, device


def synthesize(model, text: str, speaker: str, sample_rate: int, put_accent: bool, put_yo: bool):
    """Один вызов модели; длинные тексты режутся и склеиваются."""
    import torch

    parts = []
    for chunk in split_long(text):
        audio = model.apply_tts(text=chunk, speaker=speaker, sample_rate=sample_rate,
                                put_accent=put_accent, put_yo=put_yo)
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio = audio.reshape(-1).to(torch.float32)
        if not torch.isfinite(audio).all():
            raise ValueError("модель вернула нечисловые сэмплы")
        parts.append(audio)
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def write_wav(path: Path, audio, sample_rate: int, subtype: str = "PCM_16"):
    import soundfile as sf

    numpy_audio = audio.detach().cpu().numpy()
    peak = float(abs(numpy_audio).max()) if numpy_audio.size else 0.0
    if peak >= 1.0:
        raise ValueError(f"перегрузка сигнала (пик {peak:.3f}) в {path.name}")
    sf.write(str(path), numpy_audio, sample_rate, subtype=subtype)


def render(args) -> int:
    import soundfile as sf
    import torch

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    (out / "segments").mkdir(parents=True, exist_ok=True)
    book_hash = manifest["book"]["input_sha256"]
    sample_rate = args.sample_rate

    voice_by_role = {}
    for pair in (args.voice_map or "").split(","):
        if "=" in pair:
            role, voice = pair.split("=", 1)
            voice_by_role[role.strip()] = voice.strip()
    default_voice = args.speaker

    model, device = load_model(Path(args.model).expanduser(), args.model_url, args.device, args.threads)
    revision = f"silero-{Path(args.model).stem}"
    signature_base = [revision, str(device), book_hash, sample_rate, default_voice, args.temperature]

    segments = manifest["segments"]
    print(f"сегментов в манифесте: {len(segments)} | книга: {manifest['book']['title']}")

    rows, metrics, fresh, cached = [], [], 0, 0
    frames_total = 0
    synthesis_seconds = raw_seconds = 0.0
    voices_used = sorted({voice_by_role.get(s["speaker"], default_voice) for s in segments})

    with sf.SoundFile(out / "sample.wav", "w", samplerate=sample_rate, channels=1,
                      subtype="PCM_16") as montage:
        for index, segment in enumerate(segments, 1):
            speaker = voice_by_role.get(segment["speaker"], default_voice)
            if speaker not in model.speakers:
                raise SystemExit(f"нет голоса {speaker}; доступны: {', '.join(model.speakers)}")
            spoken = to_silero_marks(segment["tts_text"]) if args.stress_marks == "plus" \
                else segment["tts_text"].replace(ACUTE, "")

            wav_path = out / "segments" / (segment["id"] + ".wav")
            meta_path = wav_path.with_suffix(".json")
            signature = identity(signature_base + [segment["id"], spoken, segment["speaker"]])
            cached_ok = False
            if args.reuse and wav_path.exists() and meta_path.exists():
                previous = json.loads(meta_path.read_text(encoding="utf-8"))
                cached_ok = previous.get("signature") == signature
            if cached_ok:
                audio, spent = None, 0.0
                cached += 1
            else:
                started = time.time()
                audio = synthesize(model, spoken, speaker, sample_rate, not args.model_accents, args.put_yo)
                spent = time.time() - started
                write_wav(wav_path, audio, sample_rate)
                meta_path.write_text(json.dumps({"signature": signature, "text": segment["text"],
                                                 "speaker": segment["speaker"], "spoken_text": spoken,
                                                 "voice": speaker, "sample_rate": sample_rate},
                                                ensure_ascii=False), encoding="utf-8")
                fresh += 1

            info = sf.info(wav_path)
            frames = info.frames
            audio_start = frames_total / sample_rate
            audio_end = (frames_total + frames) / sample_rate
            if audio is None:
                block = None
            montage.write(sf.read(wav_path, dtype="float32")[0])
            frames_total += frames

            pause_ms = float(segment.get("pause_after_ms") or 0.0)
            pause_frames = round(sample_rate * pause_ms / 1000)
            if pause_frames:
                montage.write(torch.zeros(pause_frames, dtype=torch.float32).numpy())
                frames_total += pause_frames

            seconds = frames / sample_rate
            raw_seconds += seconds
            synthesis_seconds += spent
            metrics.append({"segment_id": segment["id"], "speaker": segment["speaker"], "voice": speaker,
                            "audio_seconds": round(seconds, 4), "synthesis_seconds": round(spent, 4),
                            "rtf": round(spent / seconds, 4) if seconds else None, "cached": cached_ok})
            rows.append({"segment_id": segment["id"], "paragraph_id": segment.get("paragraph_id"),
                         "speaker": segment["speaker"], "text": segment["text"], "source": segment["source"],
                         "audio_start": audio_start, "audio_end": audio_end,
                         "needs_review": segment.get("needs_review", False),
                         "tts_text": segment["tts_text"], "spoken_text": spoken})
            print(f"[{index:4}/{len(segments)}] {segment['id']:12} {segment['speaker']:10} "
                  f"{seconds:5.2f} с аудио за {spent:5.2f} с"
                  + ("  (из кэша)" if cached_ok else f"  RTF {spent / seconds:.3f}"), flush=True)

    timeline = {"segments": rows}
    (out / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")

    final_seconds = frames_total / sample_rate
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(out / "sample.wav"),
                    "-codec:a", "libmp3lame", "-b:a", "192k", str(out / "sample.mp3")], check=True)
    report = {"state": "complete", "engine": "silero-v5", "model": Path(args.model).stem,
              "model_path": str(Path(args.model).expanduser().resolve()), "model_revision": revision,
              "generation_profile": f"Silero v5, put_accent={not args.model_accents}",
              "book": manifest["book"], "sample_rate": sample_rate,
              "voice": default_voice, "voice_map": voice_by_role, "voices_used": voices_used,
              "stress_marks": args.stress_marks, "target_audio_seconds": args.target_seconds,
              "device": str(device), "device_info": {"torch_threads": args.threads},
              "platform": platform.platform(), "python": platform.python_version(),
              "packages": {p: installed_version(p) for p in ("torch", "soundfile", "numpy")},
              "model_load_seconds": 0.0, "warmup_seconds": 0.0,
              "synthesis_seconds": round(synthesis_seconds, 4),
              "raw_audio_seconds": round(raw_seconds, 4), "final_audio_seconds": final_seconds,
              "rtf": round(synthesis_seconds / raw_seconds, 4) if raw_seconds else None,
              "realtime_speed": round(raw_seconds / synthesis_seconds, 2) if synthesis_seconds else None,
              "fresh_segments": fresh, "cached_segments": cached,
              "output_files": {"wav": str(out / "sample.wav"), "mp3": str(out / "sample.mp3")},
              "segments": metrics}
    (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDONE: {final_seconds:.1f} с звука; синтез {synthesis_seconds:.1f} с; "
          f"RTF {report['rtf']:.3f} ({report['realtime_speed']:.0f}× реального времени); "
          f"свежих {fresh}, из кэша {cached}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Silero v5: синтез речи из подготовленного манифеста")
    parser.add_argument("--manifest", required=True, help="подготовленный манифест (source/manifest.json)")
    parser.add_argument("--out", required=True, help="каталог для аудио (project/audio)")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "models" / "silero_v5_ru.pt"),
                        help="файл модели v5_ru.pt (скачается, если его нет)")
    parser.add_argument("--model-url", default=MODEL_URL, help="откуда скачать модель")
    parser.add_argument("--speaker", default="aidar",
                        help="голос (v5_ru: aidar, baya, kseniya, eugene, xenia; "
                             "v5_cis_base: ru_igor, ru_roman, ru_bogdan, ru_eduard, ru_zinaida и др.)")
    parser.add_argument("--voice-map", default="", help="роли по голосам, например narrator=eugene,dialog=aidar")
    parser.add_argument("--sample-rate", type=int, default=24000, choices=[8000, 24000, 48000])
    parser.add_argument("--threads", type=int, default=4, help="потоки CPU")
    parser.add_argument("--device", default="cpu", help="cpu, cuda или mps")
    parser.add_argument("--stress-marks", choices=["plus", "strip"], default="plus",
                        help="plus — перевести наши ударения в «+» (по умолчанию), strip — убрать")
    parser.add_argument("--model-accents", action="store_true",
                        help="позволить модели ставить ударения самой (наши знаки будут проигнорированы)")
    parser.add_argument("--put-yo", default=False, action="store_true", help="автоматически ставить ё")
    parser.add_argument("--no-reuse", dest="reuse", action="store_false", help="не переиспользовать готовые сегменты")
    parser.add_argument("--temperature", type=float, default=0.0, help="входит в подпись сегментов")
    parser.add_argument("--target-seconds", type=float, default=None, help="оценка длительности (в отчёт)")
    args = parser.parse_args()
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
