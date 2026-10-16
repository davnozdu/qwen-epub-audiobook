#!/usr/bin/env -S colab run --gpu T4 --keep
# -*- coding: utf-8 -*-
"""EPUB → ZIP с аудиокнигой: book.m4b + book-read-along.epub в одном архиве.

Запуск на свежей VM Colab (одна строка, книга указывается сразу за именем файла):

    colab run --gpu T4 --keep -s book colab/qwen_tts_book.py "kniga.epub"

    # книга задаётся как: имя файла (ищется в /content, /content/drive/MyDrive, текущем каталоге),
    # любой путь внутри машины или ссылка http(s) — ссылка скачается.

Дальше файлы забираются на локальный диск:

    colab download -s book /content/out/kniga.zip ~/Downloads/

Повторный запуск на той же сессии ничего не ставит и не качает веса заново — кэш сегментов
и веса уже на машине. Каталог проекта у каждой книги свой (по имени файла), поэтому книги
не перемешиваются.

Локально (macOS, Metal) то же самое, без установки и без проверки GPU:

    python colab/qwen_tts_book.py kniga.epub --local --model-dir "<папка с весами>"

Сам конвейер не переписывается: вся работа стадий — в colab/colab_run_all.py, этот файл
только находит книгу, запускает конвейер, упаковывает результат в zip и печатает итог.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile

REPO_URL = "https://github.com/davnozdu/qwen-epub-audiobook.git"
REPO_DIR = pathlib.Path("/content/qwen-epub-audiobook")
MODEL_NAME = "Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source"
MODEL_DIR = pathlib.Path("/content/models") / MODEL_NAME
RESULT_FILES = ("book.m4b", "book-read-along.epub")


def banner(text):
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


def find_book(name):
    """Имя файла, путь или ссылка → путь к epub на этой машине."""
    if name.startswith(("http://", "https://")):
        target = pathlib.Path("/content") / pathlib.Path(name.split("?")[0]).name
        banner(f"Скачиваю книгу: {name}")
        urllib.request.urlretrieve(name, target)
        print("скачано:", target, target.stat().st_size, "байт")
        return target

    candidates = [pathlib.Path(name).expanduser()]
    if not pathlib.Path(name).is_absolute():
        here = pathlib.Path(name)
        candidates += [pathlib.Path.cwd() / here, pathlib.Path("/content") / here,
                       pathlib.Path("/content/drive/MyDrive") / here,
                       pathlib.Path("/content/drive/MyDrive/books") / here]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # последняя попытка: файл с таким именем где-нибудь под /content
    found = sorted(pathlib.Path("/content").rglob(pathlib.Path(name).name))
    if found:
        return found[0].resolve()
    raise SystemExit(
        f"книга не найдена: {name}\n"
        f"  искал: {[str(c) for c in candidates]}\n"
        "  загрузите файл на машину: colab upload -s <сессия> ~/Downloads/kniga.epub /content/kniga.epub")


def ensure_runner(repo_url, repo_dir):
    """Возвращает путь к colab_run_all.py: рядом с этим файлом или в свежем клоне.

    Когда файл исполняет Colab CLI, `__file__` в ядре не определён — это не ошибка, просто
    значит, что рядом с нами репозитория нет и его надо клонировать.
    """
    here = globals().get("__file__")
    near = pathlib.Path(here).resolve().parent / "colab_run_all.py" if here else None
    if near is not None and near.is_file():
        print("использую конвейер из этой копии репозитория:", near)
        return near

    repo_dir = pathlib.Path(repo_dir)
    banner(f"Репозиторий: {repo_url} → {repo_dir}")
    if (repo_dir / ".git").is_dir():
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "-q"], check=False)
    else:
        shutil.rmtree(repo_dir, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", repo_url, str(repo_dir)], check=True)
    runner = repo_dir / "colab" / "colab_run_all.py"
    assert runner.is_file(), f"в клоне нет {runner}"
    print("конвейер:", runner)
    return runner


def run_pipeline(args, runner, book):
    command = [sys.executable, str(runner), str(book),
               "--out", str(args.out), "--highlight", args.highlight,
               "--model-dir", str(args.model_dir)]
    if args.seconds:
        command += ["--test-seconds", str(args.seconds)]
    if args.voice:
        command += ["--voice", str(args.voice)]
    if args.local:
        command += ["--skip-install", "--skip-gpu-check"]      # на маке ничего не ставим и GPU не требуем
    banner("Конвейер: " + " ".join(str(p) for p in command))
    process = subprocess.Popen(command, cwd=str(runner.parent.parent), env=dict(os.environ),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        tail.append(line)
        del tail[:-40]
    process.wait()
    if process.returncode != 0:
        raise SystemExit("конвейер упал с кодом "
                         f"{process.returncode}\nпоследние строки:\n{''.join(tail)}")


def pack(args, book):
    result = pathlib.Path(args.out) / "project" / "result"       # так его создаёт stage_package
    missing = [name for name in RESULT_FILES if not (result / name).is_file()]
    assert not missing, f"в {result} нет файлов: {missing}"

    archive = pathlib.Path(args.zip)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in RESULT_FILES:
            zf.write(result / name, arcname=name)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    banner("Готово")
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:24} {info.file_size / 1e6:7.1f} МБ")
    print(f"\nархив      : {archive}")
    print(f"размер     : {archive.stat().st_size / 1e6:.1f} МБ")
    print(f"sha256     : {digest}")

    drive_root = pathlib.Path("/content/drive/MyDrive")
    if drive_root.is_dir():
        drive_dir = drive_root / "audiobooks"
        drive_dir.mkdir(parents=True, exist_ok=True)
        copy = drive_dir / archive.name
        shutil.copyfile(archive, copy)
        print(f"на Google Drive: {copy}")
    else:
        print("Google Drive не подключён (colab drivemount -s <сессия>) — забирайте архив командой:")

    print("\nскачать на свой диск:\n"
          f"  colab download -s <сессия> {archive} ~/Downloads/{archive.name}")
    return archive


def main():
    parser = argparse.ArgumentParser(description="EPUB → ZIP с аудиокнигой и read-along EPUB")
    parser.add_argument("book", help="имя файла, путь или ссылка на epub")
    parser.add_argument("--out", default=None,
                        help="рабочий каталог проекта (по умолчанию /content/out/<имя книги>)")
    parser.add_argument("--zip", default=None, help="путь к архиву (по умолчанию рядом с --out)")
    parser.add_argument("--seconds", type=float, default=None, help="только фрагмент такой длительности")
    parser.add_argument("--highlight", choices=["sentence", "fragment"], default="sentence",
                        help="подсветка по предложениям (точнее) или по абзацам (быстрее: меньше сегментов)")
    parser.add_argument("--voice", default=None, help="свой образец голоса вместо вшитого в репозиторий")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="где держать веса модели синтеза")
    parser.add_argument("--repo-url", default=REPO_URL, help="откуда клонировать конвейер")
    parser.add_argument("--repo-dir", default=str(REPO_DIR), help="куда клонировать конвейер")
    parser.add_argument("--local", action="store_true",
                        help="локальный прогон (macOS/Metal): без pip install и без требования GPU")
    args, extra = parser.parse_known_args()      # у ядра бывают свои аргументы (colab cli, ipykernel)
    if extra:
        print("лишние аргументы проигнорированы:", extra, flush=True)
    os.environ.setdefault("MLX_CUDA_CONV_CACHE_SIZE", "12288")     # иначе «Cache thrashing» на CUDA

    # каждому epub — свой каталог проекта: id сегментов у разных книг совпадают, кэши нельзя мешать
    stem = pathlib.Path(args.book.split("?")[0]).stem or "book"
    base = pathlib.Path("/content/out") if pathlib.Path("/content").is_dir() else pathlib.Path.cwd() / "out"
    args.out = pathlib.Path(args.out or base / stem)
    args.zip = pathlib.Path(args.zip or base / f"{stem}.zip")

    banner("Параметры запуска")
    for key, value in vars(args).items():
        print(f"  {key:10} {value}")
    print("  платформа ", sys.platform, "| python", sys.version.split()[0])

    book = find_book(args.book)
    print("\nкнига:", book, f"{book.stat().st_size / 1e6:.2f} МБ")
    runner = ensure_runner(args.repo_url, args.repo_dir)

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    run_pipeline(args, runner, book)
    pack(args, book)


if __name__ == "__main__":
    main()
