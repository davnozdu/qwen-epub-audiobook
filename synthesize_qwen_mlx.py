#!/usr/bin/env python3
"""Local Qwen3-TTS 1.7B with frozen role references and a measured Metal benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import shutil
import gc
import time
import tempfile
import signal


def installed_version(name):
    """Версия пакета или None, если его нет в этом окружении.

    На Linux нет mlx-metal (там mlx-cuda-12), и падать из-за строчки в отчёте нельзя:
    это происходит в самом конце синтеза, когда всё аудио уже сгенерировано.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

VOICE_STYLES = {
    "narrator": "Adult male narrator, warm mid-low baritone, smooth resonant timbre, measured neutral storytelling.",
    "matvey_egorovich": "Adult male, deep rounded bass, full chest resonance, relaxed confident delivery.",
    "nikolay_petrovich": "Young adult male, light bright tenor, precise crisp articulation, brisk restrained delivery.",
    "luka": "Adult male, high nimble tenor, slightly nasal bright timbre, lively quick delivery.",
    "gomozov": "Adult male, low dark baritone, dry slightly gravelly timbre, slow reserved delivery.",
    "afanasy_yagodka": "Older adult male, rough raspy mid-low voice, firm sharp articulation, sardonic delivery.",
    "arina": "Adult woman around forty. Clearly feminine female voice, medium-high pitch, soft warm light timbre. Gentle reserved delivery, clear articulation. Not a male voice or a deep baritone.",
    "sofya_ivanovna": "Adult woman. Clearly feminine female voice, bright ringing upper-mid pitch, smooth rounded timbre, confident expressive delivery. Distinct from the softer female voice of Arina. Not a male voice.",
    "group_station_people": "A single adult male voice for collective lines, bright forceful midrange, energetic clear delivery. No overlapping voices.",
}


def reference_instruction(voice):
    gender = voice.get("voice_gender", voice.get("gender"))
    unknown = gender not in {"male", "female"}
    age = str(voice.get("age", "unknown")).lower()
    child = any(word in age for word in ("child", "boy", "girl", "реб", "мальчик", "девочка"))
    casting = (("Young girl, feminine child voice." if gender == "female" else "Young boy, masculine child voice.") if child
               else ("Clearly female feminine voice." if gender == "female" else "Clearly male masculine voice."))
    if unknown:
        casting = "Neutral adult voice."       # пол роли не задан — образец всё равно нужен
    style = VOICE_STYLES.get(voice["id"], voice["voice_description"])
    return casting + " " + style + " Speak natural Russian with clear intelligible diction. Keep a single consistent speaker. Read every word exactly. No singing, laughter, whispers, sound effects or exaggerated acting."

MODEL_REVISION = "f90d617701d9f7f4ca499291e0b57f2b3c2fd2ee"
BASE_REVISION = "e7dd0585652209fa0d7783659aad4e8a324de11c"
RAB_REFERENCE_TEXT = "Сегодня тихое утро. Я спокойно и ясно читаю эту книгу."


# Группа синтеза — до 260 знаков текста книги (~20 с) — один вызов модели. Делить её по 250 значило
# бы отрезать от 255-знаковой склейки хвост «Он сел.» и снова отдать модели два слова. Текст для
# модели бывает длиннее текста книги (числа словами, «скобка открывается»), отсюда запас до 320.
RAB_ONE_CALL_CHARS = 320


def rab_chunks(text, limit=250):
    import importlib.util
    script = Path(__file__).resolve().parent / "rab/qwen_tts_stress.py"
    if not script.is_file():
        script = Path(__file__).resolve().parent.parent / "rab/qwen_tts_stress.py"
    spec = importlib.util.spec_from_file_location("rab_working_tts", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_text(module.normalize_stress(text), limit)


def clone_options(reference, args):
    """Working rab invocation: waveform input and native MLX sampler defaults."""
    if getattr(args, "rab_generation", False):
        if "ref_wave" not in reference:
            from mlx_audio.utils import load_audio
            reference["ref_wave"] = load_audio(reference["ref_audio"], sample_rate=24000)
        return dict(ref_audio=reference["ref_wave"], ref_text=reference["ref_text"],
                    lang_code="russian", temperature=args.temperature,
                    max_tokens=args.max_tokens)
    return dict(ref_audio=reference["ref_audio"], ref_text=reference["ref_text"],
                lang_code="Russian", temperature=args.temperature, max_tokens=args.max_tokens,
                top_k=50, top_p=1.0, repetition_penalty=1.05, stream=False, verbose=False)


def library_profile(voice, book_hash):
    return identity([book_hash, voice["id"], voice["instruct"]])


def read_reference(library, voice, book_hash):
    meta = Path(library) / (voice["id"] + ".json")
    if not meta.exists():
        return None
    record = json.loads(meta.read_text())
    wav = Path(library) / (voice["id"] + ".wav")
    if record["profile"] != library_profile(voice, book_hash):
        raise ValueError(f"Описание роли {voice['id']} изменилось. Выберите новую библиотеку голосов; сохранённый образец не перезаписывается автоматически.")
    if not wav.exists() or hashlib.sha256(wav.read_bytes()).hexdigest() != record["wav_sha256"]:
        raise ValueError(f"Повреждён образец голоса {voice['id']}")
    if not record.get("ref_text", "").strip():
        raise ValueError("Образец без текста")
    return dict(record, ref_audio=str(wav.resolve()))


def import_references(library, manifest, source):
    """Freeze one suitable whole utterance per role from the user's first test."""
    import soundfile as sf
    library = Path(library)
    library.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    lookup = {s["id"]: s for s in manifest["segments"]}
    book_hash = manifest["book"]["input_sha256"]
    for voice in manifest["voices"]:
        if read_reference(library, voice, book_hash):
            continue
        possible = []
        for meta in sorted((source / "segments").glob("*.json")):
            data = json.loads(meta.read_text())
            wav = meta.with_suffix(".wav")
            segment = lookup.get(meta.stem)
            if not segment or data.get("speaker") != voice["id"] or data.get("text") != segment["text"] or not wav.exists():
                continue
            if data.get("wav_sha256") != hashlib.sha256(wav.read_bytes()).hexdigest():
                continue
            info = sf.info(str(wav))
            # Avoid tiny author inserts and suspected repetitions/long silence.
            if 5 <= info.duration <= 18 and len(data["text"]) >= 60 and 7 <= len(data["text"]) / info.duration <= 25:
                possible.append((meta.stem, data, wav, info))
        if not possible:
            continue
        segment_id, data, wav, info = possible[0]
        target = library / (voice["id"] + ".wav")
        shutil.copyfile(wav, target)
        save(library / (voice["id"] + ".json"), {
            "profile": library_profile(voice, book_hash), "speaker": voice["id"],
            "name": voice["name"], "ref_text": data["text"], "source_segment": segment_id,
            "source": "frozen utterance from first VoiceDesign test",
            "audio_seconds": info.duration, "sample_rate": info.samplerate,
            "wav_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})


def complete_library(model, mx, library, manifest, args):
    import soundfile as sf
    book_hash = manifest["book"]["input_sha256"]
    for voice in manifest["voices"]:
        if read_reference(library, voice, book_hash):
            continue
        audio, sr, metrics = generate_audio(model, mx, voice["reference_text"], voice["instruct"], voice["seed"], args)
        wav = Path(library) / (voice["id"] + ".wav")
        sf.write(wav, audio, sr, subtype="PCM_16")
        save(wav.with_suffix(".json"), {
            "profile": library_profile(voice, book_hash), "speaker": voice["id"], "name": voice["name"],
            "ref_text": voice["reference_text"], "source": "VoiceDesign reference generated once",
            "audio_seconds": len(audio) / sr, "sample_rate": sr,
            "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "metrics": metrics})
        print(f"Saved permanent reference: {voice['name']}", flush=True)


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def select_segments(manifest, start_paragraph):
    start = next((i for i, s in enumerate(manifest["segments"]) if s["paragraph_id"] == start_paragraph), None)
    if start is None:
        raise ValueError(f"Нет абзаца {start_paragraph}")
    return manifest["segments"][start:]


def trim_silence(audio, sample_rate, threshold_db=-48):
    """Trim only outer silence, retaining breathing room; keep internal pauses."""
    import numpy as np
    window = max(1, int(sample_rate * 0.02))
    usable = len(audio) // window * window
    if usable == 0:
        return audio
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, window) ** 2, axis=1))
    active = np.flatnonzero(rms > 10 ** (threshold_db / 20))
    if not len(active):
        raise ValueError("Модель вернула тишину")
    first = max(0, int(active[0]) * window - int(sample_rate * 0.04))
    last = min(len(audio), (int(active[-1]) + 1) * window + int(sample_rate * 0.08))
    return audio[first:last]


def synthesis_groups(candidates):
    """Подряд идущие строки одной synthesis_group → один вызов модели; [(номер, сегмент), ...].

    Строка без группы (манифест, подготовленный до групп) — сама себе группа.
    """
    groups = []
    for index, segment in enumerate(candidates, 1):
        key = segment.get("synthesis_group")
        if groups and key is not None and groups[-1][-1][1].get("synthesis_group") == key:
            groups[-1].append((index, segment))
        else:
            groups.append([(index, segment)])
    return groups


def reference_key(reference):
    """Чем модель на самом деле клонирует голос: WAV образца и его расшифровка."""
    return (reference["wav_sha256"], reference["ref_text"]) if reference else None


def join_same_reference(groups, references, minimum, maximum):
    """Склеивает соседние группы, которые модель всё равно читает одним и тем же голосом.

    В одноголосой сборке реплики (dialog) и слова автора (narrator) — разные роли с одним
    образцом. По ролям склейка их не объединяет, и «— Да, — сказал он.» уходило бы модели
    двумя вызовами по пять-двенадцать знаков. Здесь критерий — сам образец (WAV и текст):
    если он один, клонирование звучит одинаково. Правило длины то же, что у нарезки: пока
    одна из групп короче minimum и вместе не длиннее maximum. Через главу не склеиваем.
    Строки подсветки и роли остаются прежними; на стыке монтажная пауза не нужна — её даёт
    сама модель.
    """
    def size(group):
        return len(" ".join(s["text"] for _, s in group))

    result = []
    for group in groups:
        if result:
            previous, first = result[-1][-1][1], group[0][1]
            key = reference_key(references.get(previous["speaker"]))
            if (key is not None and key == reference_key(references.get(first["speaker"]))
                    and previous["chapter_id"] == first["chapter_id"]
                    and min(size(result[-1]), size(group)) < minimum
                    and size(result[-1]) + 1 + size(group) <= maximum):
                result[-1].extend(group)
                continue
        result.append(list(group))
    for group in result:
        for _, s in group[:-1]:
            s["pause_after_ms"] = 0
    return result


def group_text(group, targets):
    """Текст группы для модели. На стыке абзацев без конечного знака ставим точку.

    Заголовок «Глава первая» или строка стиха без точки, склеенные с абзацем, иначе
    прозвучали бы одной фразой: «Глава первая Он вошёл». Точка нужна только модели —
    текст книги и разметка не меняются.
    """
    parts = []
    for position, ((_, segment), target) in enumerate(zip(group, targets)):
        following = group[position + 1][1] if position + 1 < len(group) else None
        if (following is not None and following["paragraph_id"] != segment["paragraph_id"]
                and not re.search(r"[.!?…:;,][»”\"')\]]*\s*$", target)):
            target = target.rstrip() + "."
        parts.append(target)
    return " ".join(parts)


def spoken_weight(text):
    """Доля строки в звуке группы: по буквам и цифрам, без знаков ударения и пунктуации."""
    return max(1, sum(c.isalnum() for c in text.replace("́", "")))


def split_group_audio(audio, sample_rate, weights):
    """Делит звук группы между строками по самым тихим местам рядом с ожидаемой границей.

    Ожидаемая граница — пропорционально длине текста строк (речь идёт ровно, 12–13 знаков
    в секунду); вокруг неё ищем окно с наименьшей энергией (сглаженной на 150 мс), то есть
    естественную паузу между предложениями. Куски идут подряд и в сумме дают исходный звук.
    """
    import numpy as np
    count = len(weights)
    total = len(audio)
    if count == 1:
        return [audio]
    frame = max(1, int(sample_rate * 0.01))
    frames = max(1, -(-total // frame))
    padded = np.zeros(frames * frame, dtype=np.float32)
    padded[:total] = audio
    energy = np.sqrt(np.mean(padded.reshape(frames, frame) ** 2, axis=1))
    width = 15
    smooth = np.convolve(energy, np.ones(width) / width, mode="same")
    shares = np.asarray(weights, dtype=np.float64) / float(sum(weights))
    bounds = np.cumsum(shares)[:-1] * total
    shortest = max(1, min(int(sample_rate * 0.05), total // count))
    cuts, previous = [], 0
    for k, expected in enumerate(bounds, 1):
        radius = 0.4 * min(shares[k - 1], shares[k]) * total
        radius = min(max(radius, 0.25 * sample_rate), 1.5 * sample_rate)
        low = max(previous + shortest, expected - radius)
        high = min(total - (count - k) * shortest, expected + radius)
        if high <= low:
            cut = int(min(max(expected, previous + shortest), total - (count - k) * shortest))
        else:
            first, last = int(math.ceil(low / frame)), int(high // frame)
            if last < first:
                cut = int((low + high) / 2)
            else:
                window = np.arange(first, last + 1)
                quiet = smooth[window]
                # При равной тишине — ближе к ожидаемой границе.
                best = window[np.lexsort((np.abs(window * frame - expected), quiet))[0]]
                cut = int(min(max(best * frame + frame // 2, low), high))
        cuts.append(cut)
        previous = cut
    return np.split(audio, cuts)


VOICE_FLOOR = 0.5          # ниже этого сходства с образцом — точно чужой голос
VOICE_MARGIN = 0.1         # насколько склейка может быть хуже типичной для этой книги
VOICE_MIN_SECONDS = 1.5    # на коротком звуке отпечаток голоса ненадёжен — не проверяем


class VoiceGuard:
    """Страховка от «другого голоса»: сравнивает каждую склейку с образцом.

    Модель клонирует голос по образцу, но сэмплирование иногда уводит тембр — и кусок книги звучит
    другим человеком. Эмоции и интонацию это не трогает (температура прежняя): меряем ТЕМБР тем же
    кодировщиком голоса, которым модель сама читает образец (extract_speaker_embedding), и уехавшую
    склейку синтезируем заново с другим зерном, оставляя самую похожую.

    Порог относительный: склейка хуже типичного сходства этой книги (медиана − VOICE_MARGIN) или ниже
    VOICE_FLOOR. Абсолютный порог без калибровки на настоящем звуке гонял бы пересинтез по кругу.
    """

    def __init__(self, model, mx, retries):
        self.model, self.mx, self.retries = model, mx, int(retries)
        self.references, self.seen = {}, []

    @staticmethod
    def supported(model) -> bool:
        return getattr(model, "speaker_encoder", None) is not None and hasattr(model, "extract_speaker_embedding")

    def embedding(self, audio):
        import numpy as np
        vector = np.asarray(self.model.extract_speaker_embedding(self.mx.array(np.asarray(audio, dtype=np.float32))),
                            dtype=np.float32).reshape(-1)
        return vector / (float(np.linalg.norm(vector)) + 1e-9)

    def reference(self, reference):
        key = reference["wav_sha256"]
        if key not in self.references:
            import numpy as np
            from mlx_audio.utils import load_audio
            self.references[key] = self.embedding(np.asarray(load_audio(reference["ref_audio"], sample_rate=24000)))
        return self.references[key]

    def similarity(self, audio, sample_rate, reference):
        import numpy as np
        if sample_rate != 24000 or len(audio) < sample_rate * VOICE_MIN_SECONDS:
            return None
        return float(np.dot(self.embedding(audio), self.reference(reference)))

    def threshold(self) -> float:
        import statistics
        if len(self.seen) < 5:
            return VOICE_FLOOR
        return max(VOICE_FLOOR, statistics.median(self.seen) - VOICE_MARGIN)


def guarded_generate(guard, model, mx, text, instruct, seed, args, reference):
    """Синтез склейки со страховкой голоса: уехавший тембр — пересинтез с другим зерном."""
    audio, sr, metrics = generate_audio(model, mx, text, instruct, seed, args, reference=reference)
    if guard is None or reference is None:
        return audio, sr, metrics
    score = guard.similarity(audio, sr, reference)
    if score is None:
        return audio, sr, metrics
    limit = guard.threshold()
    best, spent, attempt = (score, audio, sr, metrics), metrics["synthesis_seconds"], 0
    while best[0] < limit and attempt < guard.retries:
        attempt += 1
        print(f"голос уехал: сходство с образцом {best[0]:.2f} < {limit:.2f} — пересинтез с другим зерном "
              f"(попытка {attempt + 1})", flush=True)
        again = generate_audio(model, mx, text, instruct, seed + 7919 * attempt, args, reference=reference)
        spent += again[2]["synthesis_seconds"]
        candidate = guard.similarity(again[0], again[1], reference)
        if candidate is not None and candidate > best[0]:
            best = (candidate, *again)
    score, audio, sr, metrics = best
    # Время синтеза — честное: все попытки, иначе RTF книги выглядел бы лучше, чем был.
    metrics = dict(metrics, synthesis_seconds=spent, rtf=spent / max(metrics["raw_audio_seconds"], 1e-9),
                   realtime_speed=metrics["raw_audio_seconds"] / max(spent, 1e-9),
                   voice_similarity=round(score, 4), voice_threshold=round(limit, 4), voice_attempts=attempt + 1)
    guard.seen.append(score)
    return audio, sr, metrics


def share_metrics(metrics, part, sample_rate, group_samples, group_size):
    """Метрики строки из метрик группы: время синтеза и токены — пропорционально звуку."""
    share = len(part) / group_samples if group_samples else 1 / group_size
    result = audio_metrics(part, sample_rate)
    result.update(synthesis_seconds=metrics["synthesis_seconds"] * share,
                  raw_audio_seconds=metrics["raw_audio_seconds"] * share,
                  audio_tokens=round(metrics.get("audio_tokens", 0) * share),
                  rtf=metrics["rtf"], realtime_speed=metrics["realtime_speed"],
                  group_size=group_size, group_audio_seconds=group_samples / sample_rate,
                  group_synthesis_seconds=metrics["synthesis_seconds"])
    if metrics.get("punctuation_only"):
        result["punctuation_only"] = True
    for key in ("voice_similarity", "voice_threshold", "voice_attempts"):
        if key in metrics:
            result[key] = metrics[key]
    return result


def audio_metrics(audio, sr):
    import numpy as np
    return {"audio_seconds": len(audio) / sr, "sample_rate": sr,
            "peak": float(np.max(np.abs(audio))),
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999))}


def generate_audio(model, mx, text, instruct, seed, args, reference=None):
    import numpy as np
    if not any(c.isalnum() for c in text):
        # A punctuation-only original role span must stay anchored in EPUB,
        # but is not speech. Supply a short silent interval without calling TTS.
        sr = 24000
        audio = np.zeros(round(sr * 0.05), dtype=np.float32)
        metrics = audio_metrics(audio, sr)
        metrics.update(synthesis_seconds=0.0, raw_audio_seconds=0.05,
                       audio_tokens=0, rtf=0.0, realtime_speed=0.0, punctuation_only=True)
        return audio, sr, metrics
    mx.random.seed(seed)
    mx.synchronize()
    started = time.perf_counter()
    options = dict(temperature=args.temperature,
        max_tokens=args.max_tokens, top_k=50, top_p=1.0, repetition_penalty=1.05,
        stream=False, verbose=False)
    if reference is None:
        results = list(model.generate_voice_design(text=text, instruct=instruct, language="Russian", **options))
    else:
        limit = 250 if len(text) > RAB_ONE_CALL_CHARS else max(250, len(text))
        chunks = rab_chunks(text, limit) if getattr(args, "rab_generation", False) else [text]
        results = []
        for index, chunk in enumerate(chunks):
            mx.random.seed(seed + index)
            parts = list(model.generate(text=chunk, **clone_options(reference, args)))
            if not parts:
                raise ValueError("Модель не вернула аудио для части предложения")
            results.extend(parts)
    mx.synchronize()
    if not results:
        raise ValueError("Модель не вернула аудио")
    if any(r.token_count >= args.max_tokens for r in results):
        raise ValueError("Достигнут лимит аудиотокенов: фрагмент мог оборваться")
    rates = {r.sample_rate for r in results}
    if len(rates) != 1:
        raise ValueError("Несогласованная частота дискретизации")
    sr = rates.pop()
    audio = np.concatenate([np.asarray(r.audio, dtype=np.float32).reshape(-1) for r in results])
    elapsed = time.perf_counter() - started
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Пустое/нечисловое аудио")
    raw_duration = len(audio) / sr
    audio = trim_silence(audio, sr)
    metrics = audio_metrics(audio, sr)
    metrics.update(synthesis_seconds=elapsed, raw_audio_seconds=raw_duration,
                   audio_tokens=sum(r.token_count for r in results),
                   rtf=elapsed / raw_duration, realtime_speed=raw_duration / elapsed)
    return audio, sr, metrics


def speech_text(segment, stress_marks="keep"):
    text = segment.get("tts_text", segment["text"])
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Пустой текст озвучки")
    return text.replace("\u0301", "") if stress_marks == "strip" else text


def render(args):
    if args.ephemeral_voices:
        if args.voice_library or args.reference_source or args.import_only:
            raise ValueError("--ephemeral-voices нельзя сочетать с импортом или постоянной библиотекой")
        # Only this newly created directory is deleted, never an existing library.
        with tempfile.TemporaryDirectory(prefix="qwen-test-voices-") as folder:
            _render(args, temporary_library=folder)
        print("Temporary voice references deleted", flush=True)
    else:
        _render(args)


def _render(args, temporary_library=None):
    started = time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "segments").mkdir(exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("speech_preparation") or any("tts_text" in s for s in manifest["segments"]):
        from prepare_speech import check
        check(manifest)
    if args.ephemeral_voices:
        for voice in manifest["voices"]:
            voice["instruct"] = reference_instruction(voice)
    if getattr(args, "rab_generation", False):
        for voice in manifest["voices"]:
            voice["reference_text"] = RAB_REFERENCE_TEXT
    voices = {v["id"]: v for v in manifest["voices"]}
    candidates = manifest["segments"] if args.whole_book else select_segments(manifest, args.start_paragraph)
    model_path = Path(args.model).resolve()
    cloning = args.mode == "clone"
    revision = "rab-native-mlx-v1" if getattr(args, "rab_generation", False) else BASE_REVISION if cloning else MODEL_REVISION
    library = Path(temporary_library or args.voice_library or Path(args.manifest).resolve().parent / "voice-library")
    library.mkdir(parents=True, exist_ok=True)
    if args.reference_source:
        import_references(library, manifest, args.reference_source)
    references = {cid: read_reference(library, voice, manifest["book"]["input_sha256"]) for cid, voice in voices.items()}
    if args.import_only:
        print(json.dumps({"saved_voices": [cid for cid, ref in references.items() if ref],
                          "pending_voices": [cid for cid, ref in references.items() if not ref],
                          "library": str(library.resolve())}, ensure_ascii=False))
        return
    plan = {"state": "prepared", "model": model_path.name,
            "model_path": str(model_path), "model_revision": revision,
            "generation_profile": "rab native MLX" if getattr(args, "rab_generation", False) else "legacy",
            "book": manifest["book"], "target_audio_seconds": None if args.whole_book else args.seconds,
            "start_paragraph": candidates[0]["paragraph_id"], "temperature": args.temperature,
            "whole_book": args.whole_book,
            "max_tokens": args.max_tokens, "device_required": "Metal GPU",
            "workflow": "Frozen WAV and transcript per role; Base ICL cloning" if cloning else "Direct VoiceDesign generation",
            "voice_library": str(library.resolve()) if cloning else None,
            "temporary_references": args.ephemeral_voices,
            "voice_profiles": [{"id": v["id"], "name": v["name"], "gender": v["voice_gender"], "instruct": v["instruct"]} for v in voices.values()],
            "stop_rule": "Read every segment to the end of the manifest" if args.whole_book else "Stop after the first whole fragment reaching target audio duration; never cut a spoken fragment",
            "benchmark_definition": "RTF = synthesis seconds / raw generated audio seconds; speed = inverse RTF; model load and warmup reported separately"}
    save(out / "plan.json", plan)
    if args.plan:
        approx_chars = args.seconds * 14
        count = chars = 0
        roles = set()
        for s in candidates:
            count += 1
            chars += len(s["text"])
            roles.add(s["speaker"])
            if not args.whole_book and chars >= approx_chars:
                break
        print(json.dumps({"status": "plan_only", "estimated_segments": count,
                          "estimated_roles": sorted(roles), "characters": chars,
                          "note": "Estimate only. No audio generated and no speed measured."}, ensure_ascii=False))
        return
    # Local checkpoints and tokenizer only: no remote code, model or token downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from mlx_audio.tts.utils import load_model
    except (ImportError, RuntimeError) as exc:
        save(out / "last-error.json", dict(plan, state="blocked", audio_generated=False,
                                         error=str(exc), synthesis_seconds=None, audio_seconds=None))
        raise RuntimeError("Не удалось запустить MLX/Metal. Запустите этот тест в обычном Terminal на Mac, вне ограниченной среды выполнения. " + str(exc)) from None
    model = designer = None
    try:
        mx.set_default_device(mx.gpu)
        device_info = mx.device_info()
        mx.set_cache_limit(512 * 1024 * 1024)
        # Keep bounded allocation pressure on this 16 GB laptop. Do not change global
        # macOS GPU limits, other apps, or system power settings.
        mx.set_memory_limit(8 * 1024 ** 3)
        library_started = time.perf_counter()
        if cloning and any(ref is None for ref in references.values()):
            if not args.design_model:
                raise ValueError("Для ещё не созданных голосов укажите --design-model")
            print("Loading VoiceDesign for missing permanent references only", flush=True)
            designer = load_model(str(Path(args.design_model).resolve()))
            if designer.config.tts_model_type != "voice_design":
                raise ValueError("--design-model должен быть VoiceDesign")
            complete_library(designer, mx, library, manifest, args)
            mx.synchronize()
            del designer
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            print(f"VoiceDesign released; active MLX allocations {mx.get_active_memory() / 1e9:.3f} GB. Loading Base next.", flush=True)
            references = {cid: read_reference(library, voice, manifest["book"]["input_sha256"]) for cid, voice in voices.items()}
        library_seconds = time.perf_counter() - library_started
        load_started = time.perf_counter()
        model = load_model(str(model_path))
        mx.synchronize()
        load_seconds = time.perf_counter() - load_started
        if model.config.tts_model_type != ("base" if cloning else "voice_design") or model.speech_tokenizer is None or model.tokenizer is None:
            raise ValueError("Тип модели не соответствует --mode, либо токенизаторы не загружены")
        if cloning and not model.speech_tokenizer.has_encoder:
            raise ValueError("Для клонирования нужен аудиокодек с encoder")
        print(f"Model ready: {device_info.get('device_name', 'Metal GPU')}; load {load_seconds:.2f}s", flush=True)
        warmup_started = time.perf_counter()
        speaker = candidates[0].get("speaker")
        if speaker not in voices:
            raise ValueError(f"У первого сегмента не определён голос ({speaker!r}) — "
                             f"разметьте роли и повторите запуск")
        first_voice = voices[speaker]
        if not getattr(args, "rab_generation", False):
            generate_audio(model, mx, "Это короткая проверка звука.", first_voice["instruct"], first_voice["seed"], args,
                           reference=references[first_voice["id"]] if cloning else None)
        warmup_seconds = time.perf_counter() - warmup_started
        retries = int(getattr(args, "voice_retries", 2) or 0)
        guard = VoiceGuard(model, mx, retries) if cloning and retries > 0 and VoiceGuard.supported(model) else None
        if guard is None and cloning and retries > 0:
            print("голос: у модели нет кодировщика голоса — сходство с образцом не проверяется", flush=True)
        timeline, rows = [], []
        duration = 0.0
        fresh = 0
        sample_rate = None
        # Счётчик для прогресса: по нему бот и страница считают проценты и оценку остатка,
        # поэтому формат строки фиксированный («сегмент 412/1496 ...») и печатается на каждом
        # сегменте, включая взятые из кэша: иначе после продолжения прогона прогресс бы замер.
        total = len(candidates)
        # Единица синтеза — группа (synthesis_group): склейка коротких предложений и абзацев
        # одного голоса до ~220 знаков. Модель получает её одним вызовом, потому что на одно-
        # двухсловных кусках она рвётся; звук потом делится между строками подсветки.
        for s in candidates:
            if s["speaker"] is None or s["speaker"] not in voices:
                raise ValueError(f"Нельзя озвучить неопределённую роль {s['id']}")
        groups = synthesis_groups(candidates)
        if cloning and any("synthesis_group" in s for s in candidates):
            # Манифесты без групп (подготовленные раньше) не трогаем: их кэш остаётся годным.
            chunking = manifest.get("chunking") or {}
            groups = join_same_reference(groups, references, int(chunking.get("min_chars", 100)),
                                         int(chunking.get("max_chars", 260)))
        for group in groups:
            first = group[0][1]
            if cloning and len({reference_key(references[s["speaker"]]) for _, s in group}) != 1:
                raise ValueError(f"В группе синтеза {first.get('synthesis_group')} разные образцы голоса")
            if not cloning and len({s["speaker"] for _, s in group}) != 1:
                raise ValueError(f"В группе синтеза {first.get('synthesis_group')} смешаны голоса")
            voice = voices[first["speaker"]]
            targets = [speech_text(s, getattr(args, "stress_marks", "keep")) for _, s in group]
            reference = references[first["speaker"]] if cloning else None
            seed = voice["seed"] + len(rows) + 1 if getattr(args, "rab_generation", False) else voice["seed"]
            if len(group) == 1:
                # Подпись одиночного сегмента прежняя: кэш уже начатых книг остаётся годным.
                signatures = [identity([revision, str(model_path), manifest["book"]["input_sha256"],
                                        first["text"], targets[0], voice["instruct"], seed, args.temperature, args.max_tokens,
                                        reference["wav_sha256"] if reference else None])]
            else:
                shared = identity(["synthesis-group-1", revision, str(model_path), manifest["book"]["input_sha256"],
                                   [s["text"] for _, s in group], [s["speaker"] for _, s in group],
                                   targets, voice["instruct"], seed, args.temperature,
                                   args.max_tokens, reference["wav_sha256"] if reference else None])
                signatures = [identity([shared, position]) for position in range(len(group))]
            wavs = [out / "segments" / (s["id"] + ".wav") for _, s in group]
            pieces = None
            if not args.ephemeral_voices:
                pieces = []
                for wav, signature in zip(wavs, signatures):
                    meta = wav.with_suffix(".json")
                    previous = json.loads(meta.read_text()) if wav.exists() and meta.exists() else {}
                    if previous.get("signature") != signature or hashlib.sha256(wav.read_bytes()).hexdigest() != previous.get("wav_sha256"):
                        pieces = None
                        break
                    audio, sr = sf.read(wav, dtype="float32")
                    pieces.append((audio, sr, previous["metrics"]))
            cached = pieces is not None
            if cached and guard is not None and "voice_similarity" in pieces[0][2]:
                guard.seen.append(float(pieces[0][2]["voice_similarity"]))   # типичное сходство книги — и по кэшу
            if not cached:
                audio, sr, metrics = guarded_generate(guard, model, mx, group_text(group, targets), voice["instruct"],
                                                      seed, args, reference)
                if len(group) == 1:
                    pieces = [(audio, sr, metrics)]
                else:
                    parts = split_group_audio(audio, sr, [spoken_weight(text) for text in targets])
                    pieces = [(part, sr, share_metrics(metrics, part, sr, len(audio), len(group))) for part in parts]
                for (_, s), wav, signature, target_text, (audio, sr, metrics) in zip(group, wavs, signatures, targets, pieces):
                    sf.write(wav, audio, sr, subtype="PCM_16")
                    save(wav.with_suffix(".json"), {"signature": signature, "text": s["text"], "speaker": s["speaker"],
                                                    "spoken_text": target_text, "synthesis_group": s.get("synthesis_group"),
                                                    "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "metrics": metrics})
                fresh += len(group)
            for (index, s), target_text, (audio, sr, metrics) in zip(group, targets, pieces):
                if sample_rate is not None and sample_rate != sr:
                    raise ValueError("Частота WAV изменилась между фрагментами")
                sample_rate = sr
                offset = duration
                duration += len(audio) / sr
                timeline.append({"segment_id": s["id"], "paragraph_id": s["paragraph_id"], "speaker": s["speaker"],
                                 "text": s["text"], "source": s["source"], "audio_start": offset,
                                 "audio_end": duration, "needs_review": s["needs_review"]})
                if "tts_text" in s:
                    timeline[-1].update(tts_text=s["tts_text"], spoken_text=target_text)
                if reference:
                    timeline[-1]["reference_sha256"] = reference["wav_sha256"]
                # Outer model silence was trimmed, so the planned montage pause is added once.
                # Внутри группы пауза уже есть в звуке модели, prepare ставит там 0.
                pause_samples = round(sr * s["pause_after_ms"] / 1000)
                duration += pause_samples / sr
                rows.append(dict(metrics, segment_id=s["id"], speaker=s["speaker"], cached=cached))
                save(out / "progress.json", {"state": "generating", "audio_seconds": duration, "segments": len(rows), "last_segment": s["id"]})
                print(f"сегмент {index}/{total} · {s['id']} {voice['name']}: {metrics['audio_seconds']:.1f}s audio / {metrics['synthesis_seconds']:.1f}s generation; sample {duration:.1f}s", flush=True)
            if not args.whole_book and duration >= args.seconds:
                break
        if not timeline:
            raise ValueError("Не сгенерировано ни одного фрагмента")
        combined = out / "sample.wav"
        # Stream assembly from disk: a full book must not live in unified RAM.
        segment_by_id = {s["id"]: s for s in candidates}
        total_samples = 0
        with sf.SoundFile(combined, "w", samplerate=sample_rate, channels=1, subtype="PCM_16") as writer:
            for row in timeline:
                with sf.SoundFile(out / "segments" / (row["segment_id"] + ".wav")) as reader:
                    for block in reader.blocks(blocksize=65536, dtype="float32"):
                        writer.write(block)
                        total_samples += len(block)
                pause = round(sample_rate * segment_by_id[row["segment_id"]]["pause_after_ms"] / 1000)
                writer.write(np.zeros(pause, dtype=np.float32))
                total_samples += pause
        mp3 = out / "sample.mp3"
        # Portable preview; no tempo changes or loudness processing that hides defects.
        subprocess.run([shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(combined),
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)], check=True)
        synthesis_seconds = sum(r["synthesis_seconds"] for r in rows)
        raw_seconds = sum(r["raw_audio_seconds"] for r in rows)
        scores = [r["voice_similarity"] for r in rows if "voice_similarity" in r]
        voice_check = {"enabled": guard is not None, "checked_rows": len(scores),
                       "regenerated_rows": sum(1 for r in rows if int(r.get("voice_attempts") or 1) > 1)}
        if scores:
            import statistics
            voice_check.update(median=round(statistics.median(scores), 4), min=round(min(scores), 4))
            print(f"голос: сходство с образцом медиана {voice_check['median']:.2f}, минимум {voice_check['min']:.2f}; "
                  f"пересинтезировано строк {voice_check['regenerated_rows']}", flush=True)
        report = dict(plan, state="complete", audio_generated=True, device_info=device_info,
                      platform=platform.platform(), python=platform.python_version(),
                      packages={p: installed_version(p) for p in ["mlx", "mlx-metal", "mlx-cuda-12", "mlx-audio", "transformers", "numpy"]},
                      model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
                      voice_library_setup_seconds=library_seconds,
                      synthesis_seconds=synthesis_seconds, raw_audio_seconds=raw_seconds,
                      final_audio_seconds=total_samples / sample_rate, rtf=synthesis_seconds / raw_seconds,
                      realtime_speed=raw_seconds / synthesis_seconds,
                      model_load_warmup_synthesis_seconds=load_seconds + warmup_seconds + synthesis_seconds,
                      current_run_wall_seconds=time.perf_counter() - started,
                      fresh_segments=fresh, cached_segments=len(rows) - fresh, voice_check=voice_check,
                      peak_mlx_memory_gb=mx.get_peak_memory() / 1e9,
                      output_files={"wav": str(combined), "mp3": str(mp3)}, segments=rows)
        save(out / "timeline.json", {"segments": timeline})
        save(out / "benchmark.json", report)
        save(out / "progress.json", {"state": "complete", "audio_seconds": duration, "segments": len(rows)})
        print(f"DONE: {duration:.1f}s audio; synthesis {synthesis_seconds:.1f}s; speed {report['realtime_speed']:.2f}x realtime; {mp3}", flush=True)

    except BaseException as exc:
        # Completed traceback frames retain the model argument on failed
        # generation, even after our local 'model = None'. Drop their locals.
        import traceback
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        model = designer = None
        gc.collect()
        try:
            mx.synchronize()
            mx.clear_cache()
            remaining = mx.get_active_memory() / 1e9
            print(f"Model references cleared; active MLX allocations {remaining:.3f} GB; process exit releases remaining allocations", flush=True)
        except Exception as cleanup_error:
            print(f"GPU cleanup warning: {cleanup_error}; process exit will release remaining allocations", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True, help="local MLX Base directory for clone mode, VoiceDesign for design mode")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--whole-book", action="store_true", help="read the entire manifest, without a duration limit")
    parser.add_argument("--start-paragraph", default="p00058")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stress-marks", choices=["keep", "strip"], default="keep",
                        help="keep Unicode acute accents for an audio test; strip if this checkpoint misreads them")
    parser.add_argument("--plan", action="store_true", help="prepare and validate selection without loading MLX")
    parser.add_argument("--mode", choices=["clone", "design"], default="clone")
    parser.add_argument("--rab-generation", action="store_true", help="working rab-style native MLX generation: waveform reference and short role samples; no ICL/EOS patches")
    parser.add_argument("--voice-library", help="persistent WAV+transcript library; defaults next to manifest")
    parser.add_argument("--design-model", help="local VoiceDesign used only for missing reference voices")
    parser.add_argument("--reference-source", help="import whole voice references from a previous test directory")
    parser.add_argument("--import-only", action="store_true", help="freeze available references without GPU generation")
    parser.add_argument("--voice-retries", type=int, default=2,
                        help="сколько раз пересинтезировать склейку, если голос уехал от образца (0 — не проверять)")
    parser.add_argument("--ephemeral-voices", action="store_true", help="fresh explicit-gender references each run; delete them on exit; bypass segment cache")
    args = parser.parse_args()
    if args.seconds <= 0 or not 0 <= args.temperature <= 2 or args.max_tokens < 64:
        parser.error("Invalid seconds/temperature/token budget")
    try:
        signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
        render(args)
    except KeyboardInterrupt:
        parser.exit(130, "Тест прерван. Временные образцы очищены; процесс завершён.\n")
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
