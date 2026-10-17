#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPUB → аудиокнига (book.m4b + book-read-along.epub): весь пайплайн одним файлом.

Запуск в Colab (после того как репозиторий склонирован в /content):

    !cd /content/qwen-epub-audiobook && python colab/colab_run_all.py "/content/idiot-chapter1.epub" --test-seconds 300

Полная книга: без --test-seconds и с выходом на Google Drive:

    !cd /content/qwen-epub-audiobook && python colab/colab_run_all.py "/content/book.epub" --out /content/drive/MyDrive/audiobook

Что делает по шагам (каждый шаг печатает, что проверил, и падает с понятным текстом, если не так):

  1. env      окружение: GPU, glibc, ffmpeg
  2. deps     пины зависимостей (mlx[cuda] вместо mlx-metal и т.д.)
  3. repo     правки репозитория под Linux + словарь ударений на место
  4. model    веса Ru-Stress-CF (~4,5 ГБ) в локальную папку + проверка размера и состава
  5. manifest EPUB → манифест проекта (один чтец; с OLLAMA_API_KEY — многоголосая разметка LLM)
  6. prepare  audiobook_epub.py prepare: нарезка по предложениям + book-text.epub
  7. stress   prepare_silero.py: словарь Supertonic + Silero (ударения как в проде)
  8. voices   образец чтеца с GitHub → библиотека голосов под ПОДГОТОВЛЕННЫЙ манифест
              (именно этот хеш книги сверяет синтез, поэтому стадия идёт после prepare)
  9. synth    synthesize_qwen_mlx.py: MLX/CUDA, mode clone, rab-generation
 10. normalize normalize_book_audio.py: громкость по LUFS
 11. music    mix_book_music.py: фоновая музыка из репозитория (5 %)
 12. package  audiobook_epub.py package → book.m4b + book-read-along.epub

Повторный запуск безопасен: готовые модели, манифест, библиотека голосов и уже синтезированные
сегменты переиспользуются. Флаг --upto <шаг> останавливает на нужной стадии (например, --upto stress
или --upto synth, чтобы сначала посмотреть, что получилось).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Уведомления в Telegram — необязательные: без ключа в окружении просто ничего не отправляется.
# Модуль лежит в КОРНЕ репозитория, а sys.path[0] у скрипта — каталог colab/: подставляем корень.
REPO = Path(__file__).resolve().parent.parent          # корень репозитория
sys.path.insert(0, str(REPO))
try:
    from telegram_notify import notify as _notify
except Exception:                     # модуля может не быть в старой копии репозитория
    def _notify(text, **kwargs):      # noqa: D401
        return False


def tg(text) -> None:
    """Короткая обёртка: ошибки уведомлений не должны ломать сборку."""
    try:
        _notify(text)
    except Exception:
        pass

MODEL_REPO = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"
MODEL_NAME = "Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source"
MODEL_FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
               "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors", "speech_tokenizer/*"]
VOICE_URL = "https://raw.githubusercontent.com/davnozdu/qwen-epub-audiobook/main/resources/voices/narrator.wav"
VOICE_META_URL = "https://raw.githubusercontent.com/davnozdu/qwen-epub-audiobook/main/resources/voices/narrator.json"
VOICE_SHA256 = "fffc3115c15ebc1d6a9aac68b776ca99f92cf737e2aa99dcc1d8b8f28b2533ea"
PACKAGES = ["mlx[cuda]==0.32.3", "mlx-audio==0.5.8", "transformers==5.19.0", "huggingface-hub==1.33.0",
            "soundfile==0.14.0", "EbookLib==0.20", "lxml==6.1.3", "six==1.17.0", "razdel==0.5.0",
            "beautifulsoup4>=4.12,<5", "silero-stress==1.5"]
STAGES = ["env", "deps", "repo", "model", "manifest", "prepare", "stress", "voices",
          "synth", "normalize", "music", "package"]
# setup_audiobook.py не патчим: это локальная установка на macOS, на арендованной машине он не нужен.
PATCH_SKIP = {"setup_audiobook.py", "colab_run_all.py"}
# Кэши MLX/CUDA измеряются в ЗАПИСЯХ, а не в мегабайтах, и по умолчанию они крошечные:
# свёртки 128, граф 400, внимание 256, обратное внимание 64, FFT 128 (docs ml-explore/mlx,
# environment_variables.rst). Когда записей не хватает, MLX печатает «Cache thrashing is
# happening, please set the environment variable <имя> to a larger value than <N>» и синтез
# падает на середине книги — уже дважды: сначала на свёртках, потом на графах.
# Ставим потолки с запасом: это лимит на число записей, а не резервация памяти.
CACHE_ENTRIES = {
    "MLX_CUDA_CONV_CACHE_SIZE": 12288,
    "MLX_CUDA_GRAPH_CACHE_SIZE": 8192,
    "MLX_CUDA_SDPA_CACHE_SIZE": 4096,
    "MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE": 1024,
    "MLX_CUDA_FFT_CACHE_SIZE": 1024,
}


def _cache_env(factor):
    return {name: str(value * factor) for name, value in CACHE_ENTRIES.items()}


# Ступени на случай, если и этих потолков мало: удваиваем, а последней ступенью выключаем саму
# проверку — лучше закончить книгу с просадкой скорости, чем потерять весь прогон.
CACHE_STEPS = [_cache_env(1), _cache_env(2), _cache_env(4), {"MLX_ENABLE_CACHE_THRASHING_CHECK": "0"}]


def banner(step, text):
    print(f"\n{'=' * 72}\n[{STAGES.index(step) + 1}/{len(STAGES)}] {text}\n{'=' * 72}", flush=True)
    tg(f"⚙️[{STAGES.index(step) + 1}/{len(STAGES)}] {text}")


def run(cmd, **kwargs):
    """Запуск дочернего процесса с печатью вывода в реальном времени."""
    code, _ = run_capture(cmd, **kwargs)
    if code != 0:
        raise SystemExit(f"Команда упала с кодом {code}: {' '.join(str(p) for p in cmd)}")
    return code


def run_capture(cmd, extra_env=None, **kwargs):
    """То же, но возвращает (код, хвост вывода): по нему распознаётся Cache thrashing.

    extra_env задаёт переменные кэшей этому дочернему процессу явно, не надеясь на унаследованное
    окружение.
    """
    print("+", " ".join(str(part) for part in cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.update(extra_env or {})
    process = subprocess.Popen([str(part) for part in cmd], cwd=str(kwargs.pop("cwd", REPO)), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    tail = []
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        tail.append(line)
        del tail[:-60]
    process.wait()
    return process.returncode, "".join(tail)


# ─────────────────────────── 1. окружение ───────────────────────────
def stage_env(args):
    # Кэши MLX/CUDA: по умолчанию 128/400/256/64/128 записей, и синтез падает посреди книги
    # с «Cache thrashing is happening, please set the environment variable ... to a larger
    # value than N». Ставим до запуска дочерних процессов — они наследуют окружение.
    for name, value in CACHE_ENTRIES.items():
        os.environ.setdefault(name, str(value))
    banner("env", "Окружение: GPU, glibc, ffmpeg")
    import platform

    print("Python:", sys.version.split()[0])
    print("glibc :", ".".join(platform.libc_ver()[1].split(".")[:2]) or "?")
    print("ffmpeg:", shutil.which("ffmpeg") or "НЕ НАЙДЕН")

    gpu, why = "", []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        gpu = out.stdout.strip()
        if not gpu:
            why.append((out.stderr or "nvidia-smi вернул пусто").strip()[:120])
    except Exception as exc:
        why.append(str(exc))
    if not gpu:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu = f"{torch.cuda.get_device_name(0)} ({props.total_memory / 2 ** 30:.0f} ГБ, через torch)"
        except Exception as exc:
            why.append(f"torch: {exc}")

    if not gpu:
        try:
            import mlx.core as mx                       # MLX на Metal тоже даёт gpu
            device = mx.default_device()
            if "gpu" in str(device).lower():
                info = {}
                try:
                    info = mx.device_info()
                except Exception:
                    pass
                gpu = f"MLX {info.get('device_name') or device}"
        except Exception as exc:
            why.append(f"mlx: {exc}")
    print("GPU   :", gpu or "НЕ НАЙДЕН")
    if not gpu:
        print("  почему не нашли:", "; ".join(why) or "неизвестно")
        if not args.skip_gpu_check:
            raise SystemExit("GPU не найден. Runtime → Change runtime type → T4 GPU, затем Runtime → "
                             "Restart session и запуск заново. Без GPU синтез не поедет.")
        print("  (--skip-gpu-check: продолжаю; шаги до синтеза от GPU не зависят)")
    if "T4" not in gpu:
        print("  это не T4 — не страшно, цифры будут относиться к вашей карте.")
    print("✓ окружение готово")


# ─────────────────────────── 2. зависимости ───────────────────────────
def stage_deps(args):
    banner("deps", "Зависимости")
    if args.skip_install:
        print("--skip-install: установку пропускаю")
    else:
        run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])

    import importlib.metadata as md

    missing = []
    for module, package in (("mlx.core", "mlx"), ("mlx_audio", "mlx-audio"), ("ebooklib", "EbookLib"),
                            ("soundfile", "soundfile"), ("torch", "torch"), ("silero_stress", "silero-stress"),
                            ("numpy", "numpy"), ("razdel", "razdel")):
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f"{package} ({exc})")
    if missing:
        if args.skip_gpu_check:
            print("нет модулей (пропускаю из-за --skip-gpu-check):", "; ".join(missing))
        else:
            raise SystemExit("Не хватает зависимостей: " + "; ".join(missing))

    for package in ("mlx", "mlx-cuda-12", "mlx-audio", "silero-stress"):
        try:
            print(f"  {package:16s}", md.version(package))
        except Exception:
            print(f"  {package:16s} нет")

    try:
        import mlx.core as mx
    except Exception as exc:                       # локальный прогон без MLX: шаги до синтеза не нуждаются в нём
        if not args.skip_gpu_check:
            raise
        print(f"  MLX недоступен ({exc}) — проверку GPU пропускаю")
        print("✓ зависимости проверены без MLX")
        return
    device = mx.default_device()
    machine = str(device)
    on_gpu = "gpu" in str(device).lower()
    print("  default device  :", device, "| это GPU:", on_gpu)
    try:
        info = mx.device_info()
        machine = f"{info.get('device_name')} {info.get('architecture', '')}".strip() or str(device)
        print("  device info     :", info.get("device_name"), info.get("architecture"),
              f"{info.get('total_memory', 0) / 2 ** 30:.1f} ГБ")
    except Exception as exc:
        print("  device info     :", exc)
    if not on_gpu and not args.skip_gpu_check:
        raise SystemExit(f"MLX выбрал CPU ({device}) — синтез будет бессмысленно медленным")
    if not on_gpu:
        print("  (--skip-gpu-check: продолжаю без GPU, синтез на этой машине работать не будет)")
    print("✓ зависимости и MLX на месте")
    tg(f"🚀 Собралось начать\nкнига: {Path(args.book).name}\nмашина: {machine}")


# ─────────────────────────── 3. репозиторий ───────────────────────────
def stage_repo(args):
    banner("repo", "Правки репозитория под Linux + словарь ударений")
    changes = []
    # На macOS править нечего: пути Homebrew и mlx-metal там верные, и патч под Linux сломал бы
    # локальный прогон (ffmpeg окажется в /usr/bin, которого на маке нет).
    if sys.platform != "darwin":
        for name in ("requirements.txt", "requirements-mlx.txt"):
            path = REPO / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            patched = text.replace("mlx-metal==0.32.3", "mlx[cuda]==0.32.3")
            if patched != text:
                path.write_text(patched, encoding="utf-8")
                changes.append(f"{name}: mlx-metal → mlx[cuda]")
        # Патчим все скрипты конвейера, а не список: иначе новый файл с маковым путём роняет прогон.
        for path in sorted(REPO.glob("*.py")):
            if path.name in PATCH_SKIP:
                continue
            text = path.read_text(encoding="utf-8")
            patched = text.replace("/opt/homebrew/bin/", "/usr/bin/")
            if patched != text:
                path.write_text(patched, encoding="utf-8")
                changes.append(f"{path.name}: пути ffmpeg → /usr/bin")
    print("изменения:", changes or ("macOS: правки под Linux не нужны" if sys.platform == "darwin"
                                    else "нечего менять (уже пропатчено)"))

    if sys.platform != "darwin":
        left = [f"{p.name}:{i}" for p in sorted(REPO.glob("*.py")) if p.name != "setup_audiobook.py"
                for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                if "/opt/homebrew/bin" in line]
        assert not left, f"остались маковыe пути: {left}"

    dictionary_dir = REPO / "data" / "stress"
    dictionary_dir.mkdir(parents=True, exist_ok=True)
    target = dictionary_dir / "dictionary.json"
    source = REPO / "resources" / "stress_dictionary.json"
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
        print("словарь ударений скопирован в", target.relative_to(REPO))

    sys.path.insert(0, str(REPO))
    from stress_dictionary import StressDictionary
    dictionary = StressDictionary(str(dictionary_dir))
    print(f"словарь: {len(dictionary.data['corrections'])} слов, sha256 {dictionary.sha256[:12]}…")

    from mix_book_music import default_music
    music = default_music()
    assert music.is_file(), f"фоновая музыка не найдена: {music}"
    print("музыка:", music.name, f"{music.stat().st_size / 1e6:.1f} МБ")
    print("✓ репозиторий готов")


# ─────────────────────────── 4. модель ───────────────────────────
def stage_model(args):
    banner("model", "Веса модели синтеза (≈4,5 ГБ)")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_dir = Path(args.model_dir) if args.model_dir else Path(args.out) / "models" / MODEL_NAME
    need = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "model.safetensors", "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors"]

    def complete(path):
        if not path.is_dir():
            return False
        files = [f for f in path.rglob("*") if f.is_file() and ".cache" not in f.parts]
        size = sum(f.stat().st_size for f in files)
        return not [n for n in need if not (path / n).exists()] and size > 4e9

    if complete(model_dir):
        print("модель уже на месте:", model_dir)
    else:
        if model_dir.exists():
            print("неполный каталог — перекачиваю:", model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_REPO, local_dir=str(model_dir), allow_patterns=MODEL_FILES, max_workers=4)

    files = [f for f in Path(model_dir).rglob("*") if f.is_file() and ".cache" not in f.parts]
    size = sum(f.stat().st_size for f in files)
    missing = [n for n in need if not (Path(model_dir) / n).exists()]
    if missing:
        run(["find", str(model_dir), "-maxdepth", "2", "-type", "f"])
    assert not missing, f"не скачалось: {missing}"
    assert size > 4e9, f"веса не докачались: {size / 1e9:.2f} ГБ вместо ~4,5 ГБ"
    print(f"модель: {size / 1e9:.2f} ГБ, файлов {len(files)}")
    print("✓ модель готова")
    return str(model_dir)


# ─────────────────────────── 5. манифест ───────────────────────────
def stage_manifest(args):
    banner("manifest", "EPUB → манифест проекта")
    project = Path(args.out) / "project"
    project.mkdir(parents=True, exist_ok=True)
    manifest_path = project / "manifest.json"

    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if args.rebuild or not manifest_path.exists():
        if key:
            print("режим: многоголосый (LLM Ollama)")
            run([sys.executable, "prepare_qwen.py", "run", args.book, "-o", str(project)])
        else:
            print("режим: один чтец (без ключа; разметка проверяется валидаторами репозитория)")
            sys.path.insert(0, str(REPO))
            sys.path.insert(0, str(REPO / "colab"))
            from build_single_voice_manifest import build
            build(args.book, project)
    else:
        print("манифест уже есть:", manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("название :", manifest["book"]["title"])
    print("автор    :", manifest["book"]["author"])
    print("глав     :", len(manifest["chapters"]), "| сегментов:", len(manifest["segments"]),
          "| голосов:", len(manifest["voices"]), "| на проверку:", manifest["stats"]["review_segments"])
    for voice in manifest["voices"]:
        print(f"  {voice['id']:22s} {voice['name']}")
    print("✓ манифест готов")
    return project, manifest


# ─────────────────────────── 6. библиотека голосов ───────────────────────────
def stage_voices(args, source):
    banner("voices", "Образец чтеца с GitHub → библиотека голосов")
    import urllib.request

    from synthesize_qwen_mlx import library_profile, read_reference

    library = Path(source).parent / "voice-library"
    library.mkdir(parents=True, exist_ok=True)
    base_wav = library / "narrator.wav"
    if not base_wav.exists():
        local = Path(args.voice) if args.voice else Path("/content/narrator.wav")
        if local.is_file():
            shutil.copyfile(local, base_wav)
            print("взят локальный образец:", local)
        else:
            with urllib.request.urlopen(VOICE_URL, timeout=120) as response, base_wav.open("wb") as sink:
                shutil.copyfileobj(response, sink, 1 << 16)
            print("образец скачан:", VOICE_URL)

    digest = hashlib.sha256(base_wav.read_bytes()).hexdigest()
    print("sha256 образца:", digest)
    assert digest == VOICE_SHA256, "образец не совпал по sha256 — проверьте VOICE_URL / --voice"

    try:
        transcript = json.loads(urllib.request.urlopen(VOICE_META_URL, timeout=60).read().decode("utf-8"))["text"]
    except Exception as exc:
        transcript = ("Усмехнулся тоже и черномазый. Белокурый несколько удивился, что ему удалось "
                      "сказать довольно, впрочем, плохой каламбур.")
        print("сайдкар не скачался, беру расшифровку из скрипта:", exc)
    print("расшифровка:", transcript[:60], "…")

    # Профиль роли читаем из подготовленного манифеста: prepare перезаписал book.input_sha256 хешем
    # book-text.epub, и именно это значение synth сверяет в read_reference.
    manifest = json.loads((Path(source) / "manifest.json").read_text(encoding="utf-8"))
    book_hash = manifest["book"]["input_sha256"]
    for voice in manifest["voices"]:
        target = library / (voice["id"] + ".wav")
        if target != base_wav:
            shutil.copyfile(base_wav, target)          # реплики читает тот же тембр
        meta = {"profile": library_profile(voice, book_hash), "speaker": voice["id"], "name": voice["name"],
                "ref_text": transcript, "source": "resources/voices/narrator.wav (9,4 с)",
                "audio_seconds": 9.4, "sample_rate": 24000,
                "wav_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        (library / (voice["id"] + ".json")).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        reference = read_reference(library, voice, book_hash)
        print(f"  {voice['id']:22s} образец принят: {len(reference['ref_text'])} знаков текста")
    print("✓ голоса готовы, VoiceDesign не понадобится")
    return library


# ─────────────────────────── 7. prepare ───────────────────────────
def stage_prepare(args, project):
    banner("prepare", "Нарезка по предложениям + book-text.epub")
    source = project / "source"
    prepared = None
    if (source / "manifest.json").exists() and not args.rebuild:
        prepared = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    # Пересобираем, если нарезки нет или сменился режим подсветки: highlight_mode входит в манифест.
    if prepared is None or prepared.get("highlight_mode") != args.highlight:
        cmd = [sys.executable, "audiobook_epub.py", "prepare", "--manifest", str(project / "manifest.json"),
               "--out", str(source), "--highlight", args.highlight, "--start-paragraph", "p00001"]
        if args.test_seconds:
            cmd += ["--seconds", str(args.test_seconds)]
        run(cmd)
    else:
        print("нарезка уже есть:", source / "manifest.json")

    prepared = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    # Когда озвучивается вся книга, --seconds не задан и expected_seconds пуст: считаем оценку сами
    # (той же формулой, что и конвейер: 14 знаков в секунду плюс объявленные паузы).
    seconds = prepared["excerpt"].get("expected_seconds")
    if not seconds:
        seconds = sum(len(row["text"]) / 14 + row["pause_after_ms"] / 1000
                      for row in prepared["segments"])
    tg(f"🎧 Озвучка: {prepared['summary']['segments']} сегментов, "
       f"ожидаемое время звука ≈{seconds / 3600:.1f} ч (примерно столько же часов GPU)")
    print("сегментов на озвучку:", prepared["summary"]["segments"],
          "| абзацев:", prepared["summary"]["paragraphs"],
          "| голосов:", prepared["summary"]["voices"],
          f"| ожидаемое время звука: {seconds / 60:.0f} мин ({seconds / 3600:.1f} ч)")
    assert (source / "book-text.epub").exists(), "не собран book-text.epub"
    print("✓ нарезка готова")
    return source


# ─────────────────────────── 8. ударения ───────────────────────────
def stage_stress(args, source):
    banner("stress", "Ударения: словарь Supertonic + Silero")
    run([sys.executable, "prepare_silero.py", "--manifest", str(source / "manifest.json"),
         "--dictionary-dir", str(REPO / "data" / "stress"), "--llm-disputes", "off"])

    prepared = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    marks = sum(len(re.findall("\u0301", s.get("tts_text", ""))) for s in prepared["segments"])
    marked = sum(1 for s in prepared["segments"] if "\u0301" in s.get("tts_text", ""))
    print(f"ударений в тексте: {marks} | сегментов с ударениями: {marked} из {len(prepared['segments'])}")
    assert marks > 0, "ударения не проставились — проверьте data/stress/dictionary.json"
    sample = next(s["tts_text"] for s in prepared["segments"] if "\u0301" in s.get("tts_text", ""))
    print("пример:", sample[:110])
    print("✓ ударения расставлены")


# ─────────────────────────── 9. синтез ───────────────────────────
def stage_synth(args, source, library, model_dir):
    banner("synth", "Синтез речи (MLX/CUDA)")
    audio = source.parent / "audio"
    command = [sys.executable, "synthesize_qwen_mlx.py", "--manifest", str(source / "manifest.json"),
               "--model", model_dir, "--out", str(audio), "--mode", "clone", "--rab-generation",
               "--stress-marks", "keep", "--temperature", "0.9", "--voice-library", str(library),
               "--start-paragraph", "p00001"]
    # --whole-book всегда: границу фрагмента задаёт prepare (--seconds). Без него синтез
    # останавливается по своей оценке длительности, timeline короче манифеста и package падает.
    command.append("--whole-book")

    if args.rebuild or not (audio / "benchmark.json").exists():
        run(command + ["--plan"])
        # Ступени кэшей: при «Cache thrashing» MLX сам называет переменную и потолок, которого
        # не хватило, — повторяем с удвоенными потолками, уже синтезированные сегменты берутся
        # из кэша, работа не теряется. Последняя ступень выключает проверку совсем.
        for index, step_env in enumerate(CACHE_STEPS, 1):
            started = time.time()
            code, log = run_capture(command, extra_env=step_env)
            if code == 0:
                values = ", ".join(f"{name.split('_')[2].lower()} {value}" for name, value in step_env.items())
                print(f"синтез занял {time.time() - started:.0f} с; кэши MLX/CUDA: {values or '—'}")
                break
            missing = re.search(r"environment variable (\w+) to a larger value than (\d+)", log)
            if index < len(CACHE_STEPS):
                if missing:
                    print(f"\nсинтез упал: мало записей в {missing.group(1)} (нужно больше "
                          f"{missing.group(2)}) — повторяю со ступенью {index + 1} "
                          f"({'; '.join(f'{n}={v}' for n, v in CACHE_STEPS[index].items())})", flush=True)
                else:
                    print(f"\nсинтез упал с кодом {code} — повторяю со ступенью {index + 1}", flush=True)
                continue
            raise SystemExit(f"синтез упал с кодом {code} (кэши MLX/CUDA пройдены до последней ступени)")
    else:
        print("аудио уже собрано:", audio / "benchmark.json")

    report = json.loads((audio / "benchmark.json").read_text(encoding="utf-8"))
    print("состояние:", report.get("state"), "| звука:", round(report.get("final_audio_seconds", 0), 1), "с")
    assert report.get("state") == "complete", "прогон не завершён — запустите скрипт ещё раз, он продолжит"
    print("✓ синтез завершён")
    return audio


# ─────────────────────────── 10-11. громкость и музыка ───────────────────────────
def stage_normalize(args, audio):
    banner("normalize", "Громкость по LUFS")
    normalized = audio.parent / "audio-normalized"
    if args.rebuild or not (normalized / "benchmark.json").exists():
        run([sys.executable, "normalize_book_audio.py", "--audio-dir", str(audio), "--out", str(normalized)])
    else:
        print("уже нормализовано:", normalized)
    print("✓ громкость выровнена")
    return normalized


def stage_music(args, normalized):
    banner("music", "Фоновая музыка из репозитория (5 %)")
    with_music = normalized.parent / "audio-music"
    if args.rebuild or not (with_music / "benchmark.json").exists():
        run([sys.executable, "mix_book_music.py", "--audio-dir", str(normalized), "--out", str(with_music),
             "--volume-percent", "5"])
    else:
        print("музыка уже подмешана:", with_music)
    print("✓ музыка подмешана")
    return with_music


# ─────────────────────────── 12. упаковка ───────────────────────────
def stage_package(args, source, with_music):
    banner("package", "Сборка book.m4b + book-read-along.epub")
    result = source.parent / "result"
    run([sys.executable, "audiobook_epub.py", "package", "--manifest", str(source / "manifest.json"),
         "--audio-dir", str(with_music), "--out", str(result)])
    for name in ("book.m4b", "book-read-along.epub"):
        path = result / name
        assert path.exists(), f"нет файла {path}"
        print(f"  {name:24s} {path.stat().st_size / 1e6:8.1f} МБ  {path}")
    print("✓ готово")


def main():
    parser = argparse.ArgumentParser(description="EPUB → M4B + read-along EPUB, весь пайплайн одним файлом",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("book", help="путь к .epub")
    parser.add_argument("--out", default="/content/audiobook", help="каталог проекта (по умолчанию /content/audiobook)")
    parser.add_argument("--test-seconds", type=float, default=None,
                        help="сделать только фрагмент такой длительности, например 300")
    parser.add_argument("--model-dir", default=None, help="готовая папка с весами вместо скачивания")
    parser.add_argument("--voice", default=None, help="свой образец голоса вместо скачивания с GitHub")
    parser.add_argument("--highlight", choices=["sentence", "fragment"], default="sentence",
                        help="подсветка по предложениям (точнее) или по абзацам (меньше сегментов, быстрее)")
    parser.add_argument("--upto", choices=STAGES, default="package", help="остановиться на этой стадии")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать манифест/аудио, даже если готово")
    parser.add_argument("--skip-install", action="store_true", help="не ставить зависимости")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="не требовать GPU (для локальной проверки шагов до синтеза)")
    args = parser.parse_args()

    started = time.time()
    book = Path(args.book).expanduser().resolve()
    print("книга :", book, f"{book.stat().st_size / 1e6:.2f} МБ" if book.exists() else "НЕ НАЙДЕНА")
    assert book.is_file(), f"нет файла {book}"
    assert REPO.joinpath("synthesize_qwen_mlx.py").is_file(), f"это не корень репозитория: {REPO}"

    limit = STAGES.index(args.upto)
    project = source = library = model_dir = audio = normalized = with_music = manifest = None

    if limit >= STAGES.index("env"):
        stage_env(args)
    if limit >= STAGES.index("deps"):
        stage_deps(args)
    if limit >= STAGES.index("repo"):
        stage_repo(args)
    if limit >= STAGES.index("model"):
        model_dir = stage_model(args)
    if limit >= STAGES.index("manifest"):
        project, manifest = stage_manifest(args)
    if limit >= STAGES.index("prepare"):
        source = stage_prepare(args, project)
    if limit >= STAGES.index("stress"):
        stage_stress(args, source)
    if limit >= STAGES.index("voices"):
        # Библиотека голосов собирается ПОСЛЕ prepare: prepare перезаписывает book.input_sha256 хешем
        # собранного book-text.epub, и синтез сверяет профиль роли именно с этим значением.
        library = stage_voices(args, source)
    if limit >= STAGES.index("synth"):
        audio = stage_synth(args, source, library, model_dir)
    if limit >= STAGES.index("normalize"):
        normalized = stage_normalize(args, audio)
    if limit >= STAGES.index("music"):
        with_music = stage_music(args, normalized)
    if limit >= STAGES.index("package"):
        stage_package(args, source, with_music)

    spent_minutes = (time.time() - started) / 60
    print(f"\nвсе стадии до «{args.upto}» пройдены за {time.time() - started:.0f} с")
    if args.upto == "package":
        tg(f"✅ Готово за {spent_minutes:.0f} мин\nкнига: {Path(args.book).name}\n"
           f"файлы: book.m4b и book-read-along.epub")
        print("итог:", source.parent / "result" / "book.m4b")


if __name__ == "__main__":
    main()
