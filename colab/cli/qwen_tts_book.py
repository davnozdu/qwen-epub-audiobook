#!/usr/bin/env -S colab run --gpu T4 --keep
# -*- coding: utf-8 -*-
"""EPUB → ZIP с аудиокнигой: book.m4b + book-read-along.epub в одном архиве.

Запуск на свежей VM Colab (одна строка, книга указывается сразу за именем файла):

    colab run --gpu T4 --keep -s book colab/cli/qwen_tts_book.py "kniga.epub"

    # книга задаётся как: имя файла (ищется в /content, /content/drive/MyDrive, текущем каталоге),
    # любой путь внутри машины или ссылка http(s) — ссылка скачается.

Дальше файлы забираются на локальный диск:

    colab download -s book /content/out/kniga.zip ~/Downloads/

Повторный запуск на той же сессии ничего не ставит и не качает веса заново — кэш сегментов
и веса уже на машине. Каталог проекта у каждой книги свой (по имени файла), поэтому книги
не перемешиваются.

Локально (macOS, Metal) то же самое, без установки и без проверки GPU (веса ищутся сами,
конвейер при необходимости клонируется в ~/.cache/qwen-epub-audiobook):

    python qwen_tts_book.py kniga.epub --local

Если файл лежит внутри копии репозитория, конвейер берётся оттуда и ничего не клонируется;
если рядом репозитория нет (файл скопировали отдельно) — клонируется в ~/.cache.

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
ON_COLAB = pathlib.Path("/content").is_dir()
REPO_DIR = (pathlib.Path("/content/qwen-epub-audiobook") if ON_COLAB
            else pathlib.Path.home() / ".cache" / "qwen-epub-audiobook")
MODEL_NAME = "Qwen3-TTS-12Hz-1.7B-Ru-Stress-CF-source"
MODEL_HINTS = (pathlib.Path.home() / "Library/Application Support/Audiobook Studio/Engine/models",
               pathlib.Path.home() / ".cache/qwen-tts-models")


def default_model_dir() -> pathlib.Path:
    """Где искать веса: на Colab — /content/models, локально — знакомые места, иначе кэш."""
    if ON_COLAB:
        return pathlib.Path("/content/models") / MODEL_NAME
    for hint in MODEL_HINTS:
        if (hint / MODEL_NAME).is_dir():
            return hint / MODEL_NAME
    return pathlib.Path.home() / ".cache/qwen-tts-models" / MODEL_NAME


MODEL_DIR = default_model_dir()
RESULT_FILES = ("book.m4b", "book-read-along.epub")
DRIVE_ROOT = pathlib.Path(os.environ.get("COLAB_DRIVE_ROOT", "/content/drive/MyDrive"))


def banner(text):
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


def auto_book():
    """Книга не задана: берём единственный epub в /content или на подключённом Drive."""
    roots = [pathlib.Path("/content"), pathlib.Path("/content/drive/MyDrive")]
    found = sorted(p for root in roots if root.is_dir() for p in root.glob("*.epub"))
    if len(found) == 1:
        print("книга найдена автоматически:", found[0])
        return str(found[0])
    print("в /content найден не один epub:", [p.name for p in found])
    return ""


def find_book(name):
    """Имя файла, путь или ссылка → путь к epub на этой машине."""
    if name.startswith(("http://", "https://")):
        target = (pathlib.Path("/content") if ON_COLAB else pathlib.Path.cwd()) \
            / pathlib.Path(name.split("?")[0]).name
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
    if here:
        folder = pathlib.Path(here).resolve().parent
        for candidate in (folder / "colab_run_all.py", folder.parent / "colab_run_all.py"):
            if candidate.is_file():
                print("использую конвейер из этой копии репозитория:", candidate)
                return candidate

    repo_dir = pathlib.Path(repo_dir).expanduser()
    banner(f"Репозиторий: {repo_url} → {repo_dir}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
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

    if args.no_drive:
        print("на Google Drive не копирую: --no-drive")
    elif not DRIVE_ROOT.is_dir():
        print(f"Google Drive не подключён — папку «{args.drive_dir}» не пополняю.")
        print("  подключить (один раз на машину): colab drivemount -s <сессия>")
        print("  затем повторить задание, либо забрать архив вручную: colab download ...")
    else:
        stem = pathlib.Path(book).stem
        book_dir = DRIVE_ROOT / args.drive_dir / stem       # папка называется по имени книги
        names = {RESULT_FILES[0]: f"{stem}.m4b",
                 RESULT_FILES[1]: f"{stem}-read-along.epub"}
        try:
            book_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"Drive подключён, но создать {book_dir} не вышло: {exc}")
        else:
            for source_name in RESULT_FILES:
                target = book_dir / names[source_name]
                try:
                    shutil.copyfile(result / source_name, target)
                    print(f"на Google Drive: {target} ({target.stat().st_size / 1e6:.1f} МБ)")
                except OSError as exc:
                    print(f"не скопировалось на Drive: {target} — {exc}")

    print("\nскачать на свой диск:\n"
          f"  colab download -s <сессия> {archive} ~/Downloads/{archive.name}")
    return archive


def main():
    parser = argparse.ArgumentParser(description="EPUB → ZIP с аудиокнигой и read-along EPUB")
    parser.add_argument("book", nargs="?", default="",
                        help="имя файла, путь или ссылка на epub (можно не задавать — возьмётся единственный в /content)")
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
    parser.add_argument("--drive-dir", default="Audiobooks-colab",
                        help="папка внутри MyDrive; внутри неё — папка с именем книги")
    parser.add_argument("--no-drive", action="store_true", help="не копировать архив на Google Drive")
    parser.add_argument("--local", action="store_true",
                        help="локальный прогон (macOS/Metal): без pip install и без требования GPU")
    args, extra = parser.parse_known_args()      # у ядра бывают свои аргументы (colab cli, ipykernel)
    if extra:
        print("лишние аргументы проигнорированы:", extra, flush=True)
    # Кэши MLX/CUDA считаются в записях и по умолчанию они крошечные (свёртки 128, граф 400,
    # внимание 256/64, FFT 128): при исчерпании синтез падает на середине книги с
    # «Cache thrashing is happening, please set the environment variable ... to a larger value».
    for cache_name, cache_value in {"MLX_CUDA_CONV_CACHE_SIZE": 12288, "MLX_CUDA_GRAPH_CACHE_SIZE": 8192,
                                    "MLX_CUDA_SDPA_CACHE_SIZE": 4096,
                                    "MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE": 1024,
                                    "MLX_CUDA_FFT_CACHE_SIZE": 1024}.items():
        os.environ.setdefault(cache_name, str(cache_value))

    if not args.book:                     # так задание работает и через `colab exec -f`, где аргументы не передать
        args.book = os.environ.get("BOOK", "").strip() or auto_book()
    if not args.book:
        raise SystemExit("книга не задана: передайте имя файла или ссылку в аргументе, либо положите\n"
                         "единственный epub в /content и запустите без аргументов:\n"
                         "  colab upload -s <сессия> ~/Downloads/kniga.epub /content/kniga.epub")

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
