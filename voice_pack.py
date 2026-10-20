#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пакет голосов для многоголосой книги: образцы VoiceDesign по разметке cast.json.

Первая машина Colab: cast_book.py analyze → этот скрипт (design) → архив пакета уходит боту.
Вторая машина получает книгу и пакет и озвучивает: каждый герой клонируется из своего образца,
VoiceDesign там уже не загружается.

Пакет (папка или zip):
  cast.json            разметка абзацев и описание голосов (cast_book.py)
  voices/<id>.wav      образец героя, 24 кГц моно (рассказчика нет: он всегда вшитый narrator.wav)
  voices/<id>.json     точный текст образца (ref_text), инструкция VoiceDesign, seed, sha256 WAV

Как в audiobook-studio: образец каждого героя — его собственные реплики (~120 знаков); если
новый голос звучит как уже созданный того же пола (высота в пределах полутона, диапазон —
полтора, темп — 10 %), он создаётся заново заметно выше, затем ниже, с другим seed, и остаётся
вариант, дальше всего отстоящий от «близнеца».
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RATE = 24000
ATTEMPTS = 3                     # VoiceDesign вернул тишину или «проглотил» текст — другой seed
MAX_SAMPLE_SECONDS = 10.0        # длиннее: Base из такого образца срывается (тишина, проглоченный текст)


def log(message: str) -> None:
    print(message, flush=True)


# ─────────────────────────────── акустика ───────────────────────────────
def measure(samples, text: str):
    """Грубая просодия образца: высота (медиана, Гц), диапазон (полутона), темп (букв в секунду).

    Кадры 40 мс через 20 мс; речь — кадры не тише громких на 35 дБ; высота — нормированной
    автокорреляцией в 70–500 Гц. None — слишком короткая запись или тишина.
    """
    import numpy as np
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    window, hop, min_lag, max_lag = 960, 480, RATE // 500, RATE // 70
    if len(audio) < window + max_lag:
        return None
    starts = range(0, len(audio) - window - max_lag + 1, hop)
    energies = np.array([np.mean(audio[s:s + window] ** 2) for s in starts])
    if not len(energies):
        return None
    loud = np.sort(energies)[int((len(energies) - 1) * 0.95)]
    gate = max(1e-7, loud * 10 ** -3.5)
    pitches, active = [], 0
    for frame, energy in zip(starts, energies):
        if energy < gate:
            continue
        active += 1
        x = audio[frame:frame + window:2]
        best, scores = 0.0, {}
        for lag in range(min_lag, max_lag + 2):
            y = audio[frame + lag:frame + lag + window:2]
            scores[lag] = float(np.dot(x, y) / (math.sqrt(energy * window / 2 * float(np.dot(y, y))) + 1e-12))
        best = max(scores[l] for l in range(min_lag, max_lag + 1))
        if best > 0.6:
            for lag in range(min_lag, max_lag + 1):       # самый короткий почти лучший период
                if scores[lag] >= 0.9 * best and scores[lag] >= scores.get(lag - 1, -1) \
                        and scores[lag] >= scores[lag + 1]:
                    pitches.append(RATE / lag)
                    break
    if not active:
        return None
    seconds = active * hop / RATE
    letters = sum(1 for ch in text if ch.isalpha())
    pitches.sort()

    def quantile(p):
        return pitches[min(len(pitches) - 1, int(len(pitches) * p))]

    return {"pitch": quantile(0.5) if pitches else 0.0,
            "range": 12 * math.log2(quantile(0.9) / quantile(0.1)) if len(pitches) >= 5 else 0.0,
            "rate": letters / seconds if seconds else 0.0}


def twins(a, b) -> bool:
    """Два голоса звучат одинаково (эмбеддинг личности для этого не годится: 0,97–0,99 у всех)."""
    if not a or not b or a["pitch"] <= 0 or b["pitch"] <= 0 or a["rate"] <= 0 or b["rate"] <= 0:
        return False
    tempo = a["rate"] / b["rate"]
    return abs(12 * math.log2(a["pitch"] / b["pitch"])) < 1 and abs(a["range"] - b["range"]) < 1.5 \
        and 0.9 < tempo < 1.11


def distance(a, b) -> float:
    if not a or not b or a["pitch"] <= 0 or b["pitch"] <= 0:
        return 0.0
    return abs(12 * math.log2(a["pitch"] / b["pitch"])) + abs(a["range"] - b["range"]) / 2


def redesigned(voice: dict, higher: bool) -> str:
    """Тот же герой, заметно выше или ниже: чтобы отличался от «близнеца»."""
    import cast_book
    timbre = cast_book.sentence(voice["tts_voice"]) + (
        " Clearly higher, lighter pitch than a typical voice of this age." if higher
        else " Clearly lower, deeper pitch than a typical voice of this age.")
    return timbre + " " + cast_book.sentence(voice["tts_delivery"]) + cast_book.READING_RULES


def plausible(samples, text: str) -> str:
    """Пустая строка — образец годится; иначе причина (тишина, проглоченный или растянутый текст)."""
    import numpy as np
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(audio) or not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 1e-3:
        return "тишина"
    letters = sum(1 for ch in text if ch.isalpha())
    rate = letters / (len(audio) / RATE)
    if rate > 25:
        return f"слишком быстро ({rate:.0f} букв/с) — похоже, часть текста пропущена"
    if rate < 5:
        return f"слишком медленно ({rate:.0f} букв/с) — длинные паузы или лишние звуки"
    if len(audio) / RATE > MAX_SAMPLE_SECONDS:
        return f"образец длиннее {MAX_SAMPLE_SECONDS:g} с — из такого клонирование срывается"
    return ""


# ─────────────────────────────── создание пакета ───────────────────────────────
def design(cast_path, pack_dir, model=None, mx=None, generate=None, design_model=None) -> Path:
    """Образцы всех героев из cast.json в pack_dir/voices. Готовые (тот же текст, инструкция,
    seed и целый WAV) не пересоздаются — повторный запуск после сбоя продолжает с места."""
    import numpy as np
    import soundfile as sf
    pack = Path(pack_dir)
    voices_dir = pack / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    cast = json.loads(Path(cast_path).read_text(encoding="utf-8"))
    if Path(cast_path).resolve() != (pack / "cast.json").resolve():
        shutil.copyfile(cast_path, pack / "cast.json")
    if generate is None:
        from synthesize_qwen_mlx import generate_audio, trim_silence
        args = SimpleNamespace(temperature=0.9, max_tokens=1024)

        def generate(text, instruct, seed):
            audio, sr, _ = generate_audio(model, mx, text, instruct, seed, args)
            return np.asarray(trim_silence(audio, sr), dtype=np.float32)

        if model is None:
            import mlx.core as mx_core
            from mlx_audio.tts.utils import load_model
            mx = mx_core
            log(f"загружаю VoiceDesign: {design_model}")
            model = load_model(str(Path(design_model).resolve()))
            if model.config.tts_model_type != "voice_design":
                raise ValueError("--design-model должен быть VoiceDesign")

    def signature(voice, instruct, seed):
        return hashlib.sha256(json.dumps([voice["sample_text"], instruct, seed], ensure_ascii=False).encode()).hexdigest()

    def make(voice, instruct, seed):
        last = ""
        for attempt in range(ATTEMPTS):
            attempt_seed = seed + 1000 * attempt
            samples = generate(voice["sample_text"], instruct, attempt_seed)
            problem = plausible(samples, voice["sample_text"])
            if not problem:
                return samples, attempt_seed
            last = problem
            log(f"  образец «{voice['name']}»: {problem} — пробую другой seed")
        raise RuntimeError(f"образец «{voice['name']}» не получился за {ATTEMPTS} попытки: {last}")

    def write(voice, samples, instruct, seed, note=""):
        wav = voices_dir / f"{voice['id']}.wav"
        sf.write(wav, samples, RATE, subtype="PCM_16")
        meta = {"id": voice["id"], "name": voice["name"], "gender": voice.get("gender"),
                "ref_text": voice["sample_text"], "instruct": instruct, "seed": seed,
                "signature": signature(voice, instruct, seed), "base_signature": signature(voice, voice["instruct"],
                                                                                        voice["seed"]),
                "audio_seconds": round(len(samples) / RATE, 2),
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "note": note}
        (voices_dir / f"{voice['id']}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
        return meta

    sounds, made = {}, []
    for voice in cast["voices"]:
        if voice.get("fixed"):
            continue                                       # рассказчик — вшитый narrator.wav
        wav, meta_path = voices_dir / f"{voice['id']}.wav", voices_dir / f"{voice['id']}.json"
        if wav.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("base_signature") == signature(voice, voice["instruct"], voice["seed"]) \
                    and meta.get("sha256") == hashlib.sha256(wav.read_bytes()).hexdigest():
                log(f"голос «{voice['name']}»: образец уже есть")
                sounds[voice["id"]] = measure(sf.read(wav, dtype="float32")[0], meta["ref_text"])
                made.append(voice)
                continue
        log(f"создаю голос «{voice['name']}» ({voice.get('lines', 0)} реплик)")
        samples, seed = make(voice, voice["instruct"], voice["seed"])
        sound = measure(samples, voice["sample_text"])
        twin = next((other for other in made if other.get("gender") == voice.get("gender")
                     and voice.get("gender") in ("male", "female") and twins(sound, sounds.get(other["id"]))), None)
        note, instruct = "", voice["instruct"]
        if twin is not None:
            best = None
            for step, higher in enumerate((True, False), 1):
                candidate = redesigned(voice, higher)
                log(f"  голос похож на «{twin['name']}» — новый вариант ({'выше' if higher else 'ниже'})")
                other_samples, other_seed = make(voice, candidate, voice["seed"] + 7919 * step)
                other_sound = measure(other_samples, voice["sample_text"])
                far = distance(other_sound, sounds[twin["id"]])
                if best is None or far > best[0]:
                    best = (far, other_samples, other_seed, candidate, other_sound)
                if not twins(other_sound, sounds[twin["id"]]):
                    break
            if best and best[0] > distance(sound, sounds[twin["id"]]):
                _, samples, seed, instruct, sound = best
                note = f"создан заново: звучал как «{twin['name']}»"
                log(f"  оставлен новый вариант, отличие {best[0]:.1f} пт")
        write(voice, samples, instruct, seed, note)
        sounds[voice["id"]] = sound
        made.append(voice)
    log(f"✓ голоса готовы: {len(made)} (рассказчик — вшитый образец)")
    return pack


def zip_pack(pack_dir, target) -> Path:
    pack, target = Path(pack_dir), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(pack / "cast.json", "cast.json")
        for item in sorted((pack / "voices").glob("*")):
            if item.suffix in (".wav", ".json"):
                archive.write(item, f"voices/{item.name}")
    temp.replace(target)
    return target


def open_pack(path, folder) -> Path:
    """Папка пакета: zip распаковывается в folder, папка возвращается как есть. Проверяем, что
    у каждого героя (кроме рассказчика) есть образец с целым WAV и текстом."""
    path, folder = Path(path), Path(folder)
    if path.is_file():
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"недопустимый путь в пакете голосов: {name}")
            archive.extractall(folder)
        path = folder
    cast = json.loads((path / "cast.json").read_text(encoding="utf-8"))
    for voice in cast["voices"]:
        if voice.get("fixed"):
            continue
        meta = path / "voices" / f"{voice['id']}.json"
        wav = path / "voices" / f"{voice['id']}.wav"
        if not meta.is_file() or not wav.is_file():
            raise ValueError(f"в пакете голосов нет образца «{voice['name']}» ({voice['id']})")
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("sha256") != hashlib.sha256(wav.read_bytes()).hexdigest() or not record.get("ref_text"):
            raise ValueError(f"образец «{voice['name']}» повреждён")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cast", help="cast.json из cast_book.py analyze")
    parser.add_argument("--pack", required=True, help="папка пакета")
    parser.add_argument("--design-model", required=True, help="папка весов Qwen3-TTS VoiceDesign")
    parser.add_argument("--zip", default=None, help="куда положить архив пакета")
    args = parser.parse_args()
    design(args.cast, args.pack, design_model=args.design_model)
    if args.zip:
        log(f"пакет голосов: {zip_pack(args.pack, args.zip)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
