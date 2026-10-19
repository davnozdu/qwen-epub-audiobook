#!/usr/bin/env -S colab run --gpu T4 --keep
# -*- coding: utf-8 -*-
"""EPUB → ZIP с аудиокнигой: <название книги>.m4b + <название книги>-read-along.epub в одном архиве.

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
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
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
UNSAFE_IN_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')       # запрещены в путях Windows/Finder


def file_safe(text: str, fallback: str) -> str:
    """Название книги → имя файла: без путей, запрещённых знаков и хвостовых точек/пробелов."""
    name = re.sub(r"\s+", " ", text or "")
    name = UNSAFE_IN_NAME.sub(" - ", name).strip()          # знаки-разделители в «Дюна: Книга» не склеиваются
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .-")[:120].strip(" .-")
    return name or fallback


def result_names(args, book) -> dict:
    """Имена файлов для архива и Drive: по названию книги из EPUB, а не book.m4b.

    Название берём из манифеста проекта: его пишет prepare_qwen.py из dc:title исходного epub.
    Сначала исходный project/manifest.json — там чистое название; только если его нет, смотрим
    подготовленный project/source/manifest.json, куда audiobook_epub.py prepare уже дописал
    « — тестовый фрагмент». Название в частях книги (shrink_book.py) своё: «… — часть 1 из 3»,
    поэтому архивы частей не переименовываются одинаково.
    """
    stem = pathlib.Path(book).stem or "book"
    project = pathlib.Path(args.out) / "project"
    title = ""
    for candidate in (project / "manifest.json", project / "source" / "manifest.json"):
        try:
            title = (json.loads(candidate.read_text(encoding="utf-8")).get("book") or {}).get("title", "")
        except (OSError, ValueError):
            title = ""
        if title:
            break
    if not title:
        print(f"названия книги в манифесте нет — беру имя файла: {stem}")
    base = file_safe(title, stem)
    return {RESULT_FILES[0]: f"{base}.m4b", RESULT_FILES[1]: f"{base}-read-along.epub"}


def banner(text):
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


def notify_telegram(text_to_send: str) -> None:
    """Уведомление в Telegram, если заданы TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID. Иначе тишина."""
    try:
        import telegram_notify
    except ModuleNotFoundError:
        # telegram_notify.py лежит в корне репозитория, а не рядом с этим файлом; при запуске
        # через `colab exec` __file__ вообще не определён — берём каталог клона конвейера.
        import sys as _sys
        candidates = [pathlib.Path(globals().get("__file__", "")).resolve().parent.parent.parent,
                      REPO_DIR]
        for candidate in candidates:
            if candidate and (candidate / "telegram_notify.py").is_file():
                _sys.path.insert(0, str(candidate))
                break
        import telegram_notify
    try:
        telegram_notify.notify(text_to_send)
    except Exception:
        pass


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

    Когда файл исполняет Colab CLI, `__file__` в интерпретаторе не определён — это не ошибка,
    просто значит, что рядом с нами репозитория нет и его надо получить.
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
        # Дерево на машине правится под Linux (ffmpeg, mlx[cuda]), поэтому git pull упирается
        # в локальные изменения и клон молча остаётся старым. Берём версию из GitHub жёстко:
        # правки всё равно накладываются заново на каждом прогоне.
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "-q", "origin"], check=False)
        subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "-q", "FETCH_HEAD"], check=False)
        head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        print(f"  версия конвейера: {head or 'неизвестно'} (из GitHub)")
    else:
        shutil.rmtree(repo_dir, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", repo_url, str(repo_dir)], check=True)
    runner = repo_dir / "colab" / "colab_run_all.py"
    assert runner.is_file(), f"в клоне нет {runner}"
    print("конвейер:", runner)
    return runner


def ensure_libs_for_split(runner, book, args) -> None:
    """Нарезке нужны те же библиотеки, что и конвейеру (EbookLib, lxml, bs4).

    Раньше она падала с «No module named 'ebooklib'»: зависимости ставит первая стадия конвейера,
    а нарезка идёт до неё. Прогоняем стадию установки заранее — она потом пропустится.
    """
    if args.local or getattr(args, "skip_install", False):
        return
    probe = subprocess.run([sys.executable, "-c", "import ebooklib, lxml, bs4"],
                           capture_output=True, text=True, env=dict(os.environ))
    if probe.returncode == 0:
        return
    print("   библиотек для нарезки ещё нет — сначала стадия установки (она же нужна конвейеру)",
          flush=True)
    subprocess.run([sys.executable, str(runner), str(book), "--upto", "deps",
                    "--out", str(pathlib.Path("/content/out") / book.stem if pathlib.Path("/content").is_dir()
                                  else pathlib.Path.cwd() / "out" / book.stem)],
                   env=dict(os.environ, PYTHONPATH=str(runner.parent.parent)), check=False)


def shrink_step(runner, book, args):
    """Толстую книгу режем на части: в обработку идёт первая, остальные уезжают на Drive.

    Возвращает книгу, которую надо обрабатывать: исходную или первую часть.
    Проверка и нарезка идут на CPU до всякого синтеза, поэтому стоят минуты, а не часы.
    """
    if args.no_shrink or args.max_hours <= 0:
        return book
    repo = runner.parent.parent
    splitter = repo / "shrink_book.py"
    if not splitter.is_file():
        print(f"нарезка недоступна: в клоне нет {splitter} — обрабатываю книгу целиком")
        return book

    base = pathlib.Path("/content/out") if pathlib.Path("/content").is_dir() else pathlib.Path.cwd() / "out"
    parts_dir = base / f"{book.stem}-parts"
    banner("Нарезка: влезает ли книга в разумное время")
    ensure_libs_for_split(runner, book, args)
    command = [sys.executable, str(splitter), str(book), "--out", str(parts_dir),
               "--max-hours", str(args.max_hours), "--drive-dir", str(args.shrink_dir)]
    if args.no_drive or not DRIVE_ROOT.is_dir():
        command.append("--no-drive")          # Drive не подключён — части останутся только локально
    print("   " + " ".join(command), flush=True)
    # Вывод печатаем через ядро (как stage-скрипты в colab_run_all.py): иначе он уходит в консоль
    # машины и в терминале не появляется — тогда причина сбоя нарезки остаётся невидимой.
    process = subprocess.Popen(command, cwd=str(repo), text=True, bufsize=1,
                               env=dict(os.environ, PYTHONPATH=str(repo)),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = []
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        tail.append(line)
        del tail[:-40]
    process.wait()
    code = process.returncode

    plan_path = parts_dir / "plan.json"
    if code != 0 or not plan_path.is_file():
        print(f"нарезка не удалась (код {code}) — продолжаю с книгой целиком.")
        for line in tail[-14:]:
            print("   ", line.rstrip())
        print("  учтите: если книга длиннее сессии Colab, синтез может не дожить до конца.")
        return book
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("parts", 1) <= 1:
        notify_telegram(f"📚 «{book.stem}»: ≈{plan['estimated_hours']:.1f} ч звука — "
                        f"влезает в одну сессию, резать не надо")
        return book

    first = pathlib.Path(plan["files"][0]["epub"])
    assert first.is_file(), f"нет первой части: {first}"
    notify_telegram(f"📚 Книга большая: ≈{plan['estimated_hours']:.1f} ч звука, режу на {plan['parts']} части.\n"
                    f"В работу идёт {first.name} (≈{plan['files'][0]['estimated_hours']:.1f} ч), "
                    f"остальные — на Google Drive в папке «{args.shrink_dir}» и локально, "
                    f"их обрабатывают отдельными запусками.")
    banner("Книга большая — обрабатываю первую часть")
    print(f"  всего ≈{plan['estimated_hours']:.1f} ч звука, частей: {plan['parts']}")
    print(f"  в обработку: {first.name} (≈{plan['files'][0]['estimated_hours']:.1f} ч) — "
          f"после неё будет {first.stem}.zip")
    print(f"  остальные части: на Google Drive в папке «{args.shrink_dir}» и рядом с книгой локально;")
    print("  каждую обрабатывайте отдельным запуском:")
    for item in plan["files"][1:]:
        print(f"    {pathlib.Path(item['epub']).name}  (≈{item['estimated_hours']:.1f} ч)")
    return first


def run_pipeline(args, runner, book):
    command = [sys.executable, str(runner), str(book),
               "--out", str(args.out), "--highlight", args.highlight,
               "--polish", args.polish,
               "--model-dir", str(args.model_dir)]
    if args.upto and args.upto != "package":
        command += ["--upto", args.upto]        # режим частей останавливает конвейер на синтезе
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


DRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,}$")


def drive_link(path) -> str:
    """Ссылка на файл или папку в Google Drive.

    На смонтированном диске drivefs держит идентификатор объекта в расширенном атрибуте
    user.drive.id — по нему и собираем прямую ссылку. Если атрибута нет (старое ядро, другой
    монтировщик), возвращаем поиск по имени: он всё равно открывает нужную папку.

    Идентификатор проверяем на вид: у только что созданной папки drivefs отдаёт ВРЕМЕННУЮ
    пометку вроде «local-105», и ссылка по ней не открывается («ничего не происходит»).
    Такую пометку считаем отсутствием идентификатора.
    """
    path = pathlib.Path(path)
    file_id = ""
    try:                                    # на смонтированном Drive (Linux) атрибут читается напрямую
        file_id = os.getxattr(str(path), "user.drive.id").decode("utf-8", "replace").strip()
    except AttributeError:                  # macOS: нет os.getxattr — берём через команду xattr
        try:
            done = subprocess.run(["xattr", "-p", "user.drive.id", str(path)],
                                  capture_output=True, text=True, timeout=10,
                                  env=dict(os.environ))
            file_id = done.stdout.strip() if done.returncode == 0 else ""
        except Exception:
            file_id = ""
    except Exception:
        file_id = ""
    if file_id and DRIVE_ID_RE.match(file_id):
        if path.is_dir():
            return f"https://drive.google.com/drive/folders/{file_id}"
        return f"https://drive.google.com/file/d/{file_id}/view"
    return "https://drive.google.com/drive/search?q=" + urllib.parse.quote(path.name)


def pack(args, book):
    result = pathlib.Path(args.out) / "project" / "result"       # так его создаёт stage_package
    missing = [name for name in RESULT_FILES if not (result / name).is_file()]
    assert not missing, f"в {result} нет файлов: {missing}"

    # Файлы на диске остаются book.m4b / book-read-along.epub (их пишет audiobook_epub.py package),
    # а наружу — в архив и на Drive — уходят под названием книги из EPUB.
    names = result_names(args, book)
    archive = pathlib.Path(args.zip)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in RESULT_FILES:
            zf.write(result / name, arcname=names[name])

    # Хеш считаем потоком: read_bytes() по архиву в сотни МБ мог выбить память на машине.
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    banner("Готово")
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:24} {info.file_size / 1e6:7.1f} МБ")
    print(f"\nархив      : {archive}")
    print(f"размер     : {archive.stat().st_size / 1e6:.1f} МБ")
    print(f"sha256     : {digest}")

    drive_links = None
    if args.no_drive:
        print("на Google Drive не копирую: --no-drive")
    elif not DRIVE_ROOT.is_dir():
        print(f"Google Drive не подключён — папку «{args.drive_dir}» не пополняю.")
        print("  подключить (один раз на машину): colab drivemount -s <сессия>")
        print("  затем повторить задание, либо забрать архив вручную: colab download ...")
    else:
        stem = pathlib.Path(book).stem
        book_dir = DRIVE_ROOT / args.drive_dir / stem       # папка называется по имени файла книги
        try:
            book_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"Drive подключён, но создать {book_dir} не вышло: {exc}")
        else:
            copied = []
            for source_name in RESULT_FILES:
                target = book_dir / names[source_name]
                try:
                    shutil.copyfile(result / source_name, target)
                    print(f"на Google Drive: {target} ({target.stat().st_size / 1e6:.1f} МБ)")
                    copied.append((target.name, drive_link(target)))
                except OSError as exc:
                    print(f"не скопировалось на Drive: {target} — {exc}")
            if copied:
                drive_links = (drive_link(book_dir), copied)

    print("\nскачать на свой диск:\n"
          f"  colab download -s <сессия> {archive} ~/Downloads/{archive.name}")

    if drive_links:
        folder_link, files = drive_links
        print("\nна Google Drive:")
        print(f"  папка: {folder_link}")
        for name, link in files:
            print(f"  {name}: {link}")
        lines = "\n".join(f"• {name}: {link}" for name, link in files)
        notify_telegram("📂 Книга на Google Drive\n"
                        f"папка «{args.drive_dir}/{pathlib.Path(book).stem}»:\n{folder_link}\n{lines}")
    return archive


def pack_raw(args, book):
    """Сырьё части: только то, из чего в образе соберут книгу. Часть отдаётся КАК ЕСТЬ.

    Ни выравнивания громкости, ни фильтра, ни музыки, ни m4b здесь нет: всё это можно
    переделать без GPU, поэтому оно живёт в образе. Сегменты едут во FLAC (сжатие без потерь):
    1.5 ч это ≈260 МБ WAV против ≈110–140 МБ FLAC. sample.wav не везём вовсе — дорожку книга
    соберёт себе сама из сегментов, зато не будет второй генерации AAC-потерь.

    Манифесты нужны оба: проектный (главы, абзацы, абзацные тексты) и подготовленный
    (сегменты), плюс book-text.epub — на них опирается и сборка части, и финал.
    """
    project = pathlib.Path(args.out) / "project"
    source = project / "source"
    audio = project / "audio"
    required = [project / "manifest.json", source / "manifest.json", source / "book-text.epub",
                audio / "timeline.json", audio / "benchmark.json"]
    missing = [str(item) for item in required if not item.is_file()]
    assert not missing, f"нет сырья для части: {missing}"

    converter = shutil.which("ffmpeg")
    assert converter, "нет ffmpeg: сегменты не во что упаковать (в образе он есть, на машине поставьте ffmpeg)"
    archive = pathlib.Path(args.zip)
    archive.parent.mkdir(parents=True, exist_ok=True)
    segments = sorted((audio / "segments").glob("*.wav"))
    assert segments, f"нет сегментов в {audio / 'segments'}"

    with tempfile.TemporaryDirectory(prefix="qwen-raw-") as temporary:
        flac_dir = pathlib.Path(temporary) / "segments"
        flac_dir.mkdir()
        total = 0
        for wav in segments:
            target = flac_dir / (wav.stem + ".flac")
            done = subprocess.run([converter, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                                   "-c:a", "flac", str(target)], env=dict(os.environ))
            assert done.returncode == 0, f"не сжалось во FLAC: {wav}"
            total += target.stat().st_size
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("manifest.json", "source.json"):
                if (project / name).is_file():
                    zf.write(project / name, arcname=f"project/{name}")
            for name in ("manifest.json", "book-text.epub"):
                zf.write(source / name, arcname=f"project/source/{name}")
            # Обложка: на неё ссылаются оба манифеста, без неё сборка части (m4b) не начнётся.
            for folder, prefix in ((project, "project"), (source, "project/source")):
                cover = (json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                         .get("book", {}).get("cover") or {}).get("file")
                if cover and (folder / cover).is_file():
                    zf.write(folder / cover, arcname=f"{prefix}/{cover}")
            for name in ("timeline.json", "benchmark.json"):
                zf.write(audio / name, arcname=f"project/audio/{name}")
            for flac in sorted(flac_dir.iterdir()):
                zf.write(flac, arcname=f"project/audio/segments/{flac.name}")

    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    banner("Сырьё части готово")
    print(f"  сегментов : {len(segments)} (FLAC {total / 1e6:.1f} МБ)")
    print(f"  архив     : {archive}  ({archive.stat().st_size / 1e6:.1f} МБ)")
    print(f"  sha256    : {hasher.hexdigest()}")
    print("  громкость, фильтр, музыка и сборка — в образе: там это минуты и без GPU")
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
    parser.add_argument("--polish", choices=["soft", "off"],
                        default="off" if os.environ.get("POLISH", "1") == "0" else "soft",
                        help="мягкая обработка готовой речи (по умолчанию включена; POLISH=0 или --polish off выключает)")
    parser.add_argument("--voice", default=None, help="свой образец голоса вместо вшитого в репозиторий")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="где держать веса модели синтеза")
    parser.add_argument("--repo-url", default=REPO_URL, help="откуда клонировать конвейер")
    parser.add_argument("--repo-dir", default=str(REPO_DIR), help="куда клонировать конвейер")
    parser.add_argument("--drive-dir", default="Audiobooks-colab",
                        help="папка внутри MyDrive; внутри неё — папка с именем книги")
    parser.add_argument("--no-drive", action="store_true", help="не копировать архив на Google Drive")
    parser.add_argument("--max-hours", type=float, default=float(os.environ.get("SHRINK_HOURS", "7")),
                        help="если книга длиннее — резать на части (0 — не резать)")
    parser.add_argument("--shrink-dir", default=os.environ.get("SHRINK_DIR", "Audiobooks-shrink"),
                        help="папка на Drive, куда кладутся остальные части")
    parser.add_argument("--no-shrink", action="store_true", default=os.environ.get("SHRINK", "1") == "0",
                        help="не резать книгу на части")
    parser.add_argument("--local", action="store_true",
                        help="локальный прогон (macOS/Metal): без pip install и без требования GPU")
    parser.add_argument("--raw", action="store_true",
                        help="режим частей: остановиться на синтезе и отдать СЫРЬЁ части (FLAC), "
                             "без громкости, музыки и сборки — их делает образ")
    parser.add_argument("--part-index", default="", help="номер части (только для журнала)")
    parser.add_argument("--upto", default="package",
                        help="остановиться на стадии конвейера (в режиме частей всегда synth)")
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
    args.out = pathlib.Path(args.out) if args.out else None
    args.zip = pathlib.Path(args.zip) if args.zip else None
    base_dir = base

    banner("Параметры запуска")
    for key, value in vars(args).items():
        print(f"  {key:10} {value}")
    print("  платформа ", sys.platform, "| python", sys.version.split()[0])

    book = find_book(args.book)
    print("\nкнига:", book, f"{book.stat().st_size / 1e6:.2f} МБ")
    runner = ensure_runner(args.repo_url, args.repo_dir)

    book = shrink_step(runner, book, args)

    # каталог проекта и имя архива считаем от итоговой книги: у части это её собственное имя
    stem = pathlib.Path(book.name).stem or stem
    base = pathlib.Path("/content/out") if pathlib.Path("/content").is_dir() else pathlib.Path.cwd() / "out"
    args.out = pathlib.Path(args.out or base / stem)
    args.zip = pathlib.Path(args.zip or base / f"{stem}.zip")
    os.environ["STEM"] = stem          # обёртка по нему найдёт архив и части

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.raw:
        # Режим частей: конвейер идёт ТОЛЬКО до синтеза, наружу уходит сырьё, книга собирается в образе.
        args.upto = "synth"
        args.zip = base / f"{stem}-raw.zip"
        print(f"  части: {args.part_index or '?'} — сырьё уходит в {args.zip.name}, "
              f"Drive не участвует")
        run_pipeline(args, runner, book)
        pack_raw(args, book)
        return
    run_pipeline(args, runner, book)
    pack(args, book)


if __name__ == "__main__":
    main()
