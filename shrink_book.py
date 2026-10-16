#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Толстая книга → несколько частей-EPUB, чтобы каждая влезала в разумное время озвучки.

Зачем: синтез идёт примерно реалтайм (T4 ≈ 1,0 RTF), а бесплатная сессия Colab живёт максимум
12 часов. Книга на 20 часов в одну сессию не влезает физически, и её надо резать.

    python shrink_book.py kniga.epub --out parts --max-hours 7

Что делает:

  1. собирает манифест книги (та же стадия, что и в конвейере, без облачной LLM — это CPU);
  2. считает ожидаемую длительность: len(text)/14 + паузы (той же формулой, что в самом конвейере),
     и печатает её всегда — это и есть предварительная оценка «сколько времени займёт книга»;
  3. если получается больше --max-hours, режет книгу на части по границам абзацев и сохраняет
     каждую часть как самостоятельный EPUB: <имя>-part1.epub, <имя>-part2.epub, …;
  4. копирует части со 2-й на подключённый Google Drive в папку Audiobooks-shrink;
  5. пишет план в plan.json — обёртка по нему решает, что запускать в обработку.

Каждая часть — обычная книга для этого же конвейера: её можно скормить обёртке отдельно,
и она соберётся со своим именем (…-part2.zip, своя папка на Drive).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sys
from copy import deepcopy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "colab"))

import audiobook_epub as ep  # noqa: E402
import build_single_voice_manifest as single  # noqa: E402

CHARS_PER_SECOND = 14.0          # как в конвейере: audiobook_epub.select_excerpt
DRIVE_ROOT = pathlib.Path(os.environ.get("COLAB_DRIVE_ROOT", "/content/drive/MyDrive"))
DRIVE_DIR_DEFAULT = "Audiobooks-shrink"


def paragraph_durations(manifest) -> "list[tuple[str, float]]":
    """Длительность каждого абзаца: текст по 14 знаков в секунду плюс объявленные паузы."""
    result = []
    for paragraph_id, rows in ep.grouped(manifest["segments"], "paragraph_id").items():
        seconds = sum(len(row["text"]) / CHARS_PER_SECOND + row["pause_after_ms"] / 1000 for row in rows)
        result.append((paragraph_id, seconds))
    return result


def split_points(durations, parts: int) -> list[list[str]]:
    """Режем по абзацам на parts групп, стараясь уравнять длительность (жадный проход)."""
    total = sum(seconds for _, seconds in durations)
    target = total / parts
    groups, current, current_seconds = [], [], 0.0
    for index, (paragraph_id, seconds) in enumerate(durations):
        left_parts = parts - len(groups)
        left_paragraphs = len(durations) - index
        current.append(paragraph_id)
        current_seconds += seconds
        # закрываем группу, если добрали норму и остатка абзацев хватает на оставшиеся части
        if current_seconds >= target and left_parts > 1 and left_paragraphs > left_parts - 1:
            groups.append(current)
            current, current_seconds = [], 0.0
    if current:
        groups.append(current)
    return groups


def make_part(manifest, group, title, destination: pathlib.Path, texts: dict) -> dict:
    """Собирает EPUB одной части теми же функциями, что и конвейер (с проверками целостности)."""
    part = deepcopy(manifest)
    included = set(group)
    part["segments"] = [row for row in manifest["segments"] if row["paragraph_id"] in included]
    part["chapters"] = [chapter for chapter in manifest["chapters"]
                        if any(row["chapter_id"] == chapter["id"] for row in part["segments"])]
    part["paragraph_texts"] = {pid: texts[pid] for pid in group if pid in texts}
    part["book"] = dict(manifest["book"], title=title)
    part["voices"] = manifest.get("voices", {})
    ep.build_text_epub(part, destination)          # внутри пишет хэш и валидирует EPUB
    seconds = sum(len(row["text"]) / CHARS_PER_SECOND + row["pause_after_ms"] / 1000
                  for row in part["segments"])
    return {"part": title, "epub": str(destination), "paragraphs": len(group),
            "segments": len(part["segments"]), "estimated_seconds": round(seconds, 1),
            "estimated_hours": round(seconds / 3600, 2),
            "size_bytes": destination.stat().st_size}


def copy_to_drive(paths: list[pathlib.Path], drive_dir: str) -> list[str]:
    """Части со 2-й уезжают на Drive: обработать их можно позже, по одной."""
    if not DRIVE_ROOT.is_dir():
        return []
    target_dir = DRIVE_ROOT / drive_dir
    copied = []
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            target = target_dir / path.name
            shutil.copyfile(path, target)
            copied.append(str(target))
    except OSError as exc:
        print(f"  на Drive скопировать не вышло: {exc}")
    return copied


def build_book(book: pathlib.Path, work: pathlib.Path) -> tuple[pathlib.Path, dict, dict]:
    """Манифест книги (без LLM) + тексты абзацев."""
    manifest_path = single.build(str(book), str(work))
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    source = json.loads((work / "source.json").read_text(encoding="utf-8"))
    texts = {paragraph["id"]: paragraph["text"] for paragraph in source["paragraphs"]}
    return pathlib.Path(manifest_path), manifest, texts


def shrink(book: pathlib.Path, out: pathlib.Path, max_hours: float, parts: int | None,
           drive_dir: str, drive: bool, work: pathlib.Path | None = None) -> dict:
    book = pathlib.Path(book).expanduser().resolve()
    if not book.is_file():
        raise SystemExit(f"нет файла: {book}")
    out = pathlib.Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(work).expanduser().resolve() if work else out / "manifest-work"
    work.mkdir(parents=True, exist_ok=True)

    print(f"книга: {book.name} ({book.stat().st_size / 1e6:.2f} МБ)")
    print("собираю манифест (CPU, без GPU)…", flush=True)
    _, manifest, texts = build_book(book, work)

    durations = paragraph_durations(manifest)
    total = sum(seconds for _, seconds in durations)
    print(f"абзацев: {len(durations)} | сегментов: {len(manifest['segments'])} | "
          f"ожидаемая длительность: {total / 3600:.2f} ч ({total / 60:.0f} мин)")

    limit = max_hours * 3600
    needed = parts or max(1, math.ceil(total / limit))
    plan = {"book": str(book), "stem": book.stem, "paragraphs": len(durations),
            "segments": len(manifest["segments"]), "estimated_seconds": round(total, 1),
            "estimated_hours": round(total / 3600, 2), "max_hours": max_hours,
            "parts": needed, "out": str(out), "shrink_dir": drive_dir, "files": [], "drive": []}

    if needed <= 1:
        print(f"резать не нужно: книга ≈ {total / 3600:.2f} ч при пороге {max_hours} ч — "
              f"она пойдёт в обработку целиком")
        (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return plan

    groups = split_points(durations, needed)
    print()
    print("=" * 74)
    print(f"КНИГА БОЛЬШАЯ: ожидается ≈{total / 3600:.1f} ч, это больше порога {max_hours:g} ч.")
    print(f"Режу на {len(groups)} части по границам абзацев. В обработку идёт ПЕРВАЯ часть,")
    print("остальные сохраняются локально и уезжают на Google Drive в папку "
          f"«{drive_dir}» — каждую обрабатывайте отдельно.")
    print("=" * 74)

    files = []
    for index, group in enumerate(groups, 1):
        title = f"{manifest['book']['title']} — часть {index} из {len(groups)}"
        destination = out / f"{book.stem}-part{index}.epub"
        files.append(make_part(manifest, group, title, destination, texts))
    plan["files"] = files
    for item in files:
        print(f"  часть {item['part'].split(' — ')[-1]:>16}: {pathlib.Path(item['epub']).name} | "
              f"абзацев {item['paragraphs']:>5} | сегментов {item['segments']:>5} | "
              f"≈{item['estimated_hours']:.2f} ч | {item['size_bytes'] / 1e6:.2f} МБ")

    if drive:
        rest = [pathlib.Path(item["epub"]) for item in files[1:]]
        plan["drive"] = copy_to_drive(rest, drive_dir)
        if plan["drive"]:
            print(f"\nна Google Drive (папка {drive_dir}):")
            for path in plan["drive"]:
                print("  " + path)
        else:
            print(f"\nGoogle Drive не подключён — части лежат только в {out}")
            print(f"  подключить: colab drivemount -s <сессия>, затем повторить с --drive")
    plan["parts"] = len(groups)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    first = pathlib.Path(files[0]["epub"])
    print(f"\nв обработку: {first}")
    print(f"  ожидаемое время первой части ≈ {files[0]['estimated_hours']:.1f} ч звука "
          f"(столько же часов GPU)")
    print(f"остальные части обрабатывайте отдельно, например:")
    for item in files[1:]:
        print(f"  ./run-book.command {pathlib.Path(item['epub']).name}")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Резать толстую книгу на части для озвучки")
    parser.add_argument("book", help="исходный EPUB")
    parser.add_argument("--out", required=True, help="куда положить части")
    parser.add_argument("--max-hours", type=float, default=7.0,
                        help="порог, после которого книгу режем (по умолчанию 7 ч)")
    parser.add_argument("--parts", type=int, default=None, help="число частей (по умолчанию по порогу)")
    parser.add_argument("--drive-dir", default=DRIVE_DIR_DEFAULT, help="папка внутри MyDrive")
    parser.add_argument("--no-drive", action="store_true", help="не копировать части на Google Drive")
    parser.add_argument("--work", default=None, help="рабочий каталог манифеста")
    args = parser.parse_args()
    plan = shrink(args.book, pathlib.Path(args.out), args.max_hours, args.parts,
                  args.drive_dir, not args.no_drive, pathlib.Path(args.work) if args.work else None)
    print("\nPLAN_JSON " + json.dumps(plan, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
