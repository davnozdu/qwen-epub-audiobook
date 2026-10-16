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
import hashlib
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


def source_outline(book: pathlib.Path) -> dict:
    """Что есть в исходном EPUB: папка пакета, порядок документов и файлы внутри архива.

    Читаем ровно так же, как конвейер (prepare_qwen.read_source): по OPF идём по spine,
    берём документы XHTML в порядке чтения. Всё остальное — обложка, шрифты, стили,
    метаданные — берём из корневого файла как есть.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    from prepare_qwen import member

    with zipfile.ZipFile(book) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        opf_name = next(e.attrib["full-path"] for e in container.iter() if e.tag.endswith("}rootfile"))
        opf = ET.fromstring(archive.read(opf_name))
        items = {e.get("id"): e for e in opf.iter() if e.tag.endswith("}item")}
        documents = []
        for ref in (e for e in opf.iter() if e.tag.endswith("}itemref")):
            item = items.get(ref.get("idref"))
            if item is not None and item.get("media-type") == "application/xhtml+xml":
                documents.append({"id": item.get("id"), "href": member(opf_name, item.get("href")),
                                  "zip_href": item.get("href")})
    folder = opf_name.rsplit("/", 1)[0] if "/" in opf_name else "EPUB"
    return {"opf": opf_name, "folder": folder, "documents": documents}


def read_original(book: pathlib.Path):
    """Элементы исходника для частей: метаданные, обложка, ресурсы, оглавление."""
    from ebooklib import epub
    return epub.read_epub(str(book))


def build_part_from_source(book_path, original, outline, source_paragraphs, chapter_titles, group,
                           title, destination) -> dict:
    """Часть книги, собранная ИЗ ИСХОДНОГО EPUB: метаданные, обложка, шрифты, стили, главы.

    Документы берём из архива как есть и только выкидываем лишние листовые блоки (то же правило,
    что в конвейере: BLOCKS внутри body), поэтому разметка, сноски, эпиграфы и картинки в тексте
    остаются как в оригинале. Документы целиком за границей части в неё не попадают.
    """
    import posixpath
    import zipfile
    from copy import deepcopy

    from ebooklib import ITEM_DOCUMENT, epub
    from lxml import etree

    from prepare_qwen import BLOCKS

    included = set(group)
    kept = {}                                   # zip-путь документа → [первый блок, последний блок]
    order = []
    for paragraph in source_paragraphs:         # абзацы идут в порядке чтения
        if paragraph["id"] not in included:
            continue
        href = paragraph["source"]["href"]
        index = paragraph["source"]["block_index"]
        if href not in kept:
            kept[href] = [index, index]
            order.append(href)
        kept[href][1] = max(kept[href][1], index)

    dc = "http://purl.org/dc/elements/1.1/"
    new = epub.EpubBook()
    new.FOLDER_NAME = outline["folder"]
    new.version = "3.0"
    new.metadata = deepcopy(original.metadata)
    new.metadata.setdefault(dc, {})
    new.metadata[dc]["title"] = [(title, {})]
    new.set_title(title)

    import urllib.parse

    prefix = outline["folder"] + "/"

    def local(href):                             # путь внутри пакета, как его держит EbookLib
        """Имена в OPF, оглавлении и guide бывают с папкой пакета, с %-кодами и с #якорем."""
        clean = urllib.parse.unquote(href.split("#")[0])
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
        return clean

    resources = 0
    for item in original.get_items():            # обложка, шрифты, стили, картинки
        if item.get_type() == ITEM_DOCUMENT:
            continue
        new.add_item(deepcopy(item))
        resources += 1

    cover_pages = {local(entry["href"]) for entry in (original.guide or []) if entry.get("type") == "cover"}
    # Лицевое книги (титул, выходные данные, аннотация) — то, что стоит до первой текстовой главы:
    # в первую часть его берём, в остальные нет. Документы без своего текста внутри части
    # (например страницы-вклейки) остаются там, где стоят.
    with_text = {doc["href"] for doc in outline["documents"]
                 if any(p["source"]["href"] == doc["href"] for p in source_paragraphs)}
    positions = {doc["href"]: index for index, doc in enumerate(outline["documents"])}
    front_matter = min((positions[href] for href in with_text), default=0)

    with zipfile.ZipFile(book_path) as archive:
        kept_names, documents = set(), []
        for position, doc in enumerate(outline["documents"]):
            href, name = doc["href"], local(doc["href"])
            if href in kept:
                start, end = kept[href]
                soup = etree.fromstring(archive.read(href))
                bodies = soup.findall(".//{*}body") or [soup]
                body = bodies[0]
                blocks = [element for element in body.iter()
                          if isinstance(element.tag, str) and etree.QName(element).localname in BLOCKS
                          and not any(isinstance(child.tag, str) and etree.QName(child).localname in BLOCKS
                                      for child in element.iterdescendants())]
                for index, element in enumerate(blocks):
                    if not (start <= index <= end):
                        element.getparent().remove(element)
                content = etree.tostring(soup, xml_declaration=True, encoding="utf-8")
                chapter_id = next((p["chapter_id"] for p in source_paragraphs
                                   if p["source"]["href"] == href), "")
            elif name in cover_pages:
                content = archive.read(href)     # страница обложки — в каждую часть
                chapter_id = ""
            elif position < front_matter:
                content = archive.read(href)     # лицевое книги — только в первую часть
                chapter_id = ""
            else:
                continue
            page = epub.EpubHtml(uid=doc["id"] or f"doc{position:03d}", file_name=name,
                                 title=chapter_titles.get(chapter_id, ""), lang=original.language or "ru")
            page.content = content
            new.add_item(page)
            kept_names.add(name)
            documents.append(page)

    # Обложку помечаем как cover-image: часть должна показывать её в читалках, как оригинал.
    cover_href = next((local(entry["href"]) for entry in (original.guide or [])
                       if entry.get("type") == "cover"), None)
    if cover_href:
        cover_item = new.get_item_with_href(cover_href)
        if cover_item is not None:
            properties = list(getattr(cover_item, "properties", []) or [])
            if "cover-image" not in properties:
                properties.append("cover-image")
            cover_item.properties = properties

    new.spine = documents or ["nav"]
    new.guide = [entry for entry in (original.guide or []) if local(entry["href"]) in kept_names]

    counter = {"n": 0}

    def keep_toc(entries):
        """Оглавление оставляем только для тех глав, что есть в этой части."""
        result = []
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                # Вложенность разворачиваем: у части книги оглавление приходит парами
                # (пункт, [подпункты]), а писатель EbookLib ждёт ровно пару — плоский список надёжнее.
                result.extend(keep_toc(entry))
                continue
            href = getattr(entry, "href", "")
            if not href or posixpath.normpath(local(href)) not in kept_names:
                continue
            if getattr(entry, "uid", None) is None:          # NCX требует идентификатор у пункта
                counter["n"] += 1
                entry = epub.Link(href, getattr(entry, "title", ""), f"toc{counter['n']:04d}")
            result.append(entry)
        return result

    source_toc = original.toc or []
    if not isinstance(source_toc, (list, tuple)):     # у части книг ToC — один пункт, а не список
        source_toc = [source_toc]
    native = keep_toc(source_toc)
    # Если EbookLib не разобрал родное оглавление (бывает у EPUB2), строим его по главам части:
    # заголовок берём из манифеста (он читается из h1–h6 самого документа).
    generated = []
    for position, doc in enumerate(outline["documents"]):
        name = local(doc["href"])
        if name not in kept_names or name in cover_pages:
            continue
        chapter_id = next((p["chapter_id"] for p in source_paragraphs if p["source"]["href"] == doc["href"]), "")
        title = chapter_titles.get(chapter_id) or posixpath.basename(name)
        generated.append(epub.Link(name, title, f"toc{position:04d}"))
    new.toc = tuple(native) if native else tuple(generated)
    new.add_item(epub.EpubNav(title="Оглавление"))
    new.add_item(epub.EpubNcx())
    epub.write_epub(str(destination), new, {"raise_exceptions": True, "epub3_pages": False})
    return {"resources": resources, "documents": len(documents), "chapters": len(new.toc),
            "size_bytes": destination.stat().st_size}


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
    """Запасной путь: часть из нормализованного текста конвейера (без обложки и разметки)."""
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


def make_part_smart(book, manifest, source, group, index, total, out, texts) -> dict:
    """Собирает часть из исходного EPUB; если исходник необычный — откатывается на текст конвейера."""
    title = f"{manifest['book']['title']} — часть {index} из {total}"
    destination = out / f"{pathlib.Path(book).stem}-part{index}.epub"
    try:
        outline = source_outline(book)
        original = read_original(book)
        chapter_titles = {chapter["id"]: chapter["title"] for chapter in manifest.get("chapters", [])}
        details = build_part_from_source(book, original, outline, source["paragraphs"], chapter_titles,
                                         group, title, destination)
        print(f"  часть {index}: обложка и разметка из оригинала — документов {details['documents']}, "
              f"глав в оглавлении {details['chapters']}, ресурсов {details['resources']}")
    except Exception as exc:                     # необычный EPUB — не срываем нарезку
        print(f"  часть {index}: не вышло собрать из оригинала ({type(exc).__name__}: {exc}) — "
              f"собираю из нормализованного текста")
        return make_part(manifest, group, title, destination, texts)
    seconds = sum(len(row["text"]) / CHARS_PER_SECOND + row["pause_after_ms"] / 1000
                  for row in manifest["segments"] if row["paragraph_id"] in set(group))
    return {"part": title, "epub": str(destination), "paragraphs": len(group),
            "segments": sum(1 for row in manifest["segments"] if row["paragraph_id"] in set(group)),
            "estimated_seconds": round(seconds, 1), "estimated_hours": round(seconds / 3600, 2),
            "size_bytes": destination.stat().st_size, "from_source": True}


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
    """Манифест книги (без LLM) + абзацы исходника (с привязкой к блокам в XHTML)."""
    manifest_path = single.build(str(book), str(work))
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    source = json.loads((work / "source.json").read_text(encoding="utf-8"))
    return pathlib.Path(manifest_path), manifest, source


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
    _, manifest, source = build_book(book, work)
    texts = {paragraph["id"]: paragraph["text"] for paragraph in source["paragraphs"]}

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

    files = [make_part_smart(book, manifest, source, group, index, len(groups), out, texts)
             for index, group in enumerate(groups, 1)]
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
