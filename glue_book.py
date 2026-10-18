#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""glue_book.py — собрать книгу из частей: сырьё частей в томе → m4b и read-along.epub.

Работает В КОНТЕЙНЕРЕ, без облака и без машины Colab: всё, что можно переделать без GPU
(склейка, громкость, фильтр, музыка, сборка), живёт здесь и переделывается за минуты.

    # часть как самостоятельная книга (сырьё части лежит в part-2/)
    python3 glue_book.py --mode part --part 2 --parts-dir /data/parts/kniga --out /data/results
    # финал, когда готовы все части
    python3 glue_book.py --mode final --parts-dir /data/parts/kniga --book /data/books/kniga.epub \\
                         --out /data/results

Что делает финал:
 1. склеивает проектные манифесты частей в манифест всей книги. Главы, абзацы и сегменты
    перенумеровываются: внутри части ids начинаются заново с ch001/p00001, вместе они
    сталкиваются, а подсветка и главы адресуются именно этими ids;
 2. прогоняет ту же подготовку, что и на машине (`audiobook_epub.py prepare`), — получаются
    манифест всей книги и book-text.epub, на которые потом опирается сборка;
 3. СВЕРЯЕТ СТЫК: озвученные сегменты частей по порядку должны совпасть с манифестом всей
    книги (текст и роль). Не совпало — остановка с внятной причиной: книга с разъехавшейся
    подсветкой хуже, чем честная остановка;
 4. собирает аудио всей книги: сегменты частей переименовываются в ids всей книги и ложатся
    в один каталог, времена в timeline сдвигаются на сумму длительностей предыдущих частей;
 5. выравнивает громкость (+ мягкий фильтр) и накладывает музыку ОДНИМ прогоном на всю книгу:
    скачков на стыках частей не бывает по построению;
 6. собирает m4b и read-along.epub, кладёт архив с исходным названием книги в --out.

Сырьё частей никуда не девается: громкость, фильтр или музыку можно переделать и собрать
книгу заново, не трогая облако и не платя GPU-часами.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "colab" / "cli"))

import audiobook_epub                                    # noqa: E402
from qwen_tts_book import file_safe                      # noqa: E402  правило имён одно на весь конвейер

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
RESULT_FILES = ("book.m4b", "book-read-along.epub")


def load(path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def save(path, payload: dict) -> None:
    pathlib.Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


# ───────────────────────────── части на диске ─────────────────────────────

def part_dirs(parts_dir) -> list:
    """Части по номерам. Дырки в нумерации — это ошибка: склеивать книгу с пропуском нельзя."""
    found = []
    for folder in sorted(pathlib.Path(parts_dir).glob("part-*")):
        try:
            index = int(folder.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        found.append((index, folder))
    found.sort()
    return found


def check_parts(parts_dir, expect: int = 0) -> list:
    """Проверяет, что части на месте и полные, и возвращает их описание."""
    parts = []
    for index, folder in part_dirs(parts_dir):
        item = {"index": index, "folder": folder,
                "project": folder / "project" / "manifest.json",
                "source_json": folder / "project" / "source.json",
                "source": folder / "project" / "source" / "manifest.json",
                "audio": folder / "project" / "audio",
                "timeline": folder / "project" / "audio" / "timeline.json",
                "benchmark": folder / "project" / "audio" / "benchmark.json",
                "epub": folder / "project" / "source" / "book-text.epub"}
        missing = [name for name in ("source", "timeline", "benchmark", "epub")
                   if not item[name].is_file()]
        if missing:
            raise SystemExit(f"часть {index} неполная: нет {', '.join(missing)} в {folder}")
        parts.append(item)
    if not parts:
        raise SystemExit(f"в {parts_dir} нет ни одной части (ожидались part-1, part-2, …)")
    if [item["index"] for item in parts] != list(range(1, len(parts) + 1)):
        raise SystemExit("нумерация частей с дыркой: "
                         + ", ".join(str(item["index"]) for item in parts))
    if expect and len(parts) != expect:
        raise SystemExit(f"частей {len(parts)}, а книга резалась на {expect} — сборку не начинаю")
    return parts


def raw_audio(part) -> pathlib.Path:
    """Каталог с сырьём части: сегменты + timeline + benchmark (у части он один)."""
    return part["audio"]


def segments_in(audio_dir) -> dict:
    """Сегменты по id: во FLAC (так их везём из облака) или в WAV (если сырьё уже распаковано)."""
    found = {}
    for path in sorted((pathlib.Path(audio_dir) / "segments").iterdir()):
        if path.suffix.lower() in (".flac", ".wav"):
            found.setdefault(path.stem, path)
    return found


def build_sample(audio_dir, destination, total: float) -> None:
    """Собирает дорожку книги из сегментов по timeline: то же, что сделал синтез на машине.

    Сегменты кладутся на свои места (audio_start … audio_end), промежутки заполняются тишиной —
    так паузы между фразами остаются ровно там, где их посчитал синтез. Пишем потоком:
    1.5 ч это 130 млн отсчётов, в память такое целиком не берём.
    """
    import numpy as np
    import soundfile as sf

    audio_dir = pathlib.Path(audio_dir)
    rows = load(audio_dir / "timeline.json")["segments"]
    files = segments_in(audio_dir)
    info = sf.info(str(next(iter(files.values()))))
    rate = info.samplerate
    frame_count = round(total * rate)
    with sf.SoundFile(str(destination), "w", samplerate=rate, channels=1, subtype="PCM_16") as writer:
        cursor = 0
        for row in rows:
            begin = round(float(row["audio_start"]) * rate)
            if begin < cursor:
                raise SystemExit(f"сегменты накладываются друг на друга: {row['segment_id']}")
            while cursor < begin:                        # пауза между фразами
                block = min(65536, begin - cursor)
                writer.write(np.zeros(block, dtype="float32"))
                cursor += block
            source = files.get(row["segment_id"])
            if source is None:
                raise SystemExit(f"нет сегмента {row['segment_id']} в {audio_dir / 'segments'}")
            with sf.SoundFile(str(source)) as reader:
                for block in reader.blocks(blocksize=65536, dtype="float32"):
                    writer.write(block)
                    cursor += len(block)
        while cursor < frame_count:                      # хвост дорожки
            block = min(65536, frame_count - cursor)
            writer.write(np.zeros(block, dtype="float32"))
            cursor += block
    actual = sf.info(str(destination))
    if abs(actual.duration - total) > 2 / actual.samplerate:
        raise SystemExit(f"дорожка книги вышла {actual.duration:.3f} с вместо {total:.3f} с")


# ───────────────────────── склейка манифестов частей ─────────────────────────

def book_meta(book) -> dict:
    """Название и автор из исходного epub: имя файла части («… — часть 1 из 2») тут не годится."""
    title, author = "", ""
    if book and pathlib.Path(book).is_file():
        try:
            import ebooklib                                  # noqa: F401  проверка, что книга читается
            from ebooklib import epub
            original = epub.read_epub(str(book))
            title = " ".join((original.get_metadata("DC", "title") or [("", {})])[0][0].split())
            author = " ".join((original.get_metadata("DC", "creator") or [("", {})])[0][0].split())
        except Exception as exc:                         # книга без метаданных — не повод не собирать
            print(f"  метаданные не прочитались ({exc}) — беру название из части")
    return {"title": title, "author": author}


def strip_part_suffix(title: str) -> str:
    marker = " — часть "
    return title.split(marker)[0].strip() if marker in title else title.strip()


def assemble_project(parts: list, meta: dict) -> dict:
    """Проектный манифест всей книги из манифестов частей.

    Внутри каждой части главы, абзацы и сегменты нумеруются заново с ch001/p00001 — вместе они
    сталкиваются, поэтому перенумеровываем: к прежнему id спереди приписывается номер части
    (c01ch001, p0100001). Порядок частей задаёт порядок книги, а сам текст сверяется отдельно.
    """
    chapters, segments = [], []
    texts = {}
    for part in parts:
        project = load(part["project"])
        prefix = f"c{part['index']:02d}"
        chapter_map = {}
        for chapter in project.get("chapters") or []:
            fresh = prefix + str(chapter["id"])
            chapter_map[chapter["id"]] = fresh
            chapters.append(dict(chapter, id=fresh))
        paragraph_map = {}
        for row in project["segments"]:
            old = str(row["paragraph_id"])
            if old not in paragraph_map:
                paragraph_map[old] = f"p{part['index']:02d}{old.lstrip('p')}"
        for row in project["segments"]:
            fresh = deepcopy(row)
            fresh["id"] = f"s{part['index']:02d}-{row['id']}"
            fresh["chapter_id"] = chapter_map.get(row["chapter_id"], str(row["chapter_id"]))
            fresh["paragraph_id"] = paragraph_map[row["paragraph_id"]]
            segments.append(fresh)
        if part["source_json"].is_file():
            for paragraph in load(part["source_json"]).get("paragraphs") or []:
                if paragraph["id"] in paragraph_map:
                    texts[paragraph_map[paragraph["id"]]] = paragraph["text"]
        print(f"  часть {part['index']}: глав {len(project.get('chapters') or [])}, "
              f"сегментов-абзацев {len(project['segments'])}", flush=True)
    book = dict(next((item for item in (load(part["project"]) for part in parts)), {}).get("book") or {})
    book["title"] = meta.get("title") or strip_part_suffix(book.get("title") or "")
    book["author"] = meta.get("author") or book.get("author") or ""
    voices, seen = [], set()
    for part in parts:
        for voice in load(part["project"]).get("voices") or []:
            if voice["id"] not in seen:
                seen.add(voice["id"])
                voices.append(voice)
    return {"book": book, "chapters": chapters, "voices": voices, "segments": segments,
            "paragraph_texts": texts}


def verify_seam(whole: dict, parts: list) -> int:
    """Стык: озвученное в частях по порядку должно совпасть с манифестом всей книги.

    Проверяем текст и роль каждого сегмента. Это ловит и потерянную часть, и часть от другой
    книги, и разошедшуюся разметку ролей: во всех этих случаях подсветка и главы поехали бы.
    """
    rows = []
    for part in parts:
        prepared = load(part["source"])
        rows.extend(prepared["segments"])
    if len(rows) != len(whole["segments"]):
        raise SystemExit(f"стык не сходится: в частях {len(rows)} сегментов, "
                         f"а в книге {len(whole['segments'])} — часть потеряна или лишняя")
    for index, (expected, actual) in enumerate(zip(whole["segments"], rows)):
        for field in ("text", "speaker"):
            if expected.get(field) != actual.get(field):
                raise SystemExit(
                    f"стык не сходится на сегменте {index + 1} из {len(rows)} "
                    f"(поле «{field}»):\n  в книге: {str(expected.get(field))[:120]!r}\n"
                    f"  в части: {str(actual.get(field))[:120]!r}\n"
                    "Собирать книгу с разъехавшейся подсветкой нельзя: "
                    "части озвучивались не по этому тексту")
    return len(rows)


# ───────────────────────────── склейка аудио ─────────────────────────────

def merge_audio(whole: dict, parts: list, work) -> float:
    """Один каталог аудио на всю книгу: сегменты частей в ids книги и сдвинутые времена."""
    import soundfile as sf

    target = pathlib.Path(work) / "audio"
    (target / "segments").mkdir(parents=True, exist_ok=True)
    ids = [row["id"] for row in whole["segments"]]
    rows, position, total, copied = [], 0, 0.0, 0
    for part in parts:
        timeline = load(part["timeline"])["segments"]
        files = segments_in(raw_audio(part))
        shift = total
        for row in timeline:
            whole_row = whole["segments"][position]
            # Описательные поля берём из манифеста всей книги: времена — из части (со сдвигом),
            # а текст, роль и привязка к тексту должны быть книжными. Иначе сборка честно
            # отказывается работать: timeline ссылался бы на оглавление части.
            fresh = dict(row)
            fresh.update({key: value for key, value in whole_row.items()
                          if key not in ("id", "order")})
            fresh["segment_id"] = ids[position]
            fresh["audio_start"] = round(float(row["audio_start"]) + shift, 6)
            fresh["audio_end"] = round(float(row["audio_end"]) + shift, 6)
            rows.append(fresh)
            source = files.get(row["segment_id"])
            if source is None:
                raise SystemExit(f"в части {part['index']} нет сегмента {row['segment_id']}")
            destination = target / "segments" / (ids[position] + ".wav")
            if not destination.exists():                 # повторный прогон не копирует заново
                with sf.SoundFile(str(source)) as reader, sf.SoundFile(
                        str(destination), "w", samplerate=reader.samplerate, channels=1,
                        subtype="PCM_16") as writer:
                    for block in reader.blocks(blocksize=65536, dtype="float32"):
                        writer.write(block)
                copied += 1
            position += 1
        total += float(load(part["benchmark"])["final_audio_seconds"])
        print(f"  часть {part['index']}: {len(timeline)} сегментов, {shift:.1f} → {total:.1f} с",
              flush=True)
    if position != len(ids):
        raise SystemExit(f"сегментов в частях {position}, а в книге {len(ids)}")
    save(target / "timeline.json", {"segments": rows, "glue": {"parts": len(parts)}})
    save(target / "benchmark.json", {"state": "complete", "final_audio_seconds": total,
                                     "book": {"input_sha256": whole["book"]["input_sha256"]},
                                     "glued": {"parts": len(parts), "copied_segments": copied,
                                               "method": "части склеены демоном в контейнере"}})
    build_sample(target, target / "sample.wav", total)
    return total


def run_step(title: str, command: list) -> None:
    banner(title)
    print("  " + " ".join(str(item) for item in command), flush=True)
    process = subprocess.run([str(item) for item in command], cwd=str(REPO))
    if process.returncode != 0:
        raise SystemExit(f"шаг «{title}» упал с кодом {process.returncode}")


def pack_result(result, out, base: str, title: str) -> pathlib.Path:
    """Архив как у машины: имя — как у книги на диске, файлы внутри названы книгой.

    Имя архива специально без пробелов и как имя файла книги (stem): по абсолютному пути в журнале
    бот понимает, что архив готов. Название книги с пробелами внутри архива — а не в его имени.
    """
    result = pathlib.Path(result)
    missing = [name for name in RESULT_FILES if not (result / name).is_file()]
    if missing:
        raise SystemExit(f"в {result} нет файлов: {missing}")
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{base}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(result / "book.m4b", arcname=f"{title}.m4b")
        zf.write(result / "book-read-along.epub", arcname=f"{title}-read-along.epub")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    print(f"архив  : {archive}  ({archive.stat().st_size / 1e6:.1f} МБ)")
    print(f"sha256 : {digest.hexdigest()}")
    return archive


def rewrite_paths(part) -> dict:
    """Манифест части писался на машине: путь к book-text.epub там свой, у нас он другой.

    Хеш текста (input_sha256) при этом не меняется — он и есть проверка, что озвучен именно
    этот текст. А вот подпись подготовки речи (speech_preparation.input_sha256) считается в том
    числе по пути к книге, поэтому после переноса её надо пересчитать: без этого сборка части
    отказывается работать («подготовка речи устарела после изменения текста»), хотя текст тот же.
    """
    prepared = load(part["source"])
    local = str(part["epub"].resolve())
    if prepared["book"].get("epub_path") != local:
        prepared["book"]["epub_path"] = local
        preparation = prepared.get("speech_preparation") or {}
        if preparation.get("version"):
            import prepare_speech
            preparation["input_sha256"] = prepare_speech.input_identity(prepared, preparation["version"])
            print(f"  часть {part['index']}: путь к тексту перенесён, подпись подготовки речи пересчитана")
        save(part["source"], prepared)
    digest = hashlib.sha256(part["epub"].read_bytes()).hexdigest()
    if digest != prepared["book"].get("input_sha256"):
        raise SystemExit(f"часть {part['index']}: текст части изменился после озвучки "
                         f"(sha256 не совпал) — собирать её заново нельзя")
    return prepared


# ───────────────────────────────── режимы ─────────────────────────────────

def mode_part(parts: list, work, out, name: str = "") -> int:
    """Часть как самостоятельная книга: сырьё как есть — без громкости, фильтра и музыки."""
    if len(parts) != 1:
        raise SystemExit("--part требует ровно одну часть: передайте --parts-dir с part-<k>")
    part = parts[0]
    prepared = rewrite_paths(part)
    total = float(load(part["benchmark"])["final_audio_seconds"])
    raw = work / "part-raw"
    raw.mkdir(parents=True, exist_ok=True)
    banner(f"Часть {part['index']}: дорожка из сырья")
    for name_file in ("timeline.json", "benchmark.json"):
        shutil.copyfile(raw_audio(part) / name_file, raw / name_file)
    build_sample(raw_audio(part), raw / "sample.wav", total)
    banner(f"Часть {part['index']}: сборка книги части")
    run_step(f"Часть {part['index']}: m4b и read-along",
             [sys.executable, REPO / "audiobook_epub.py", "package", "--manifest", part["source"],
              "--audio-dir", raw, "--out", work / "result"])
    archive = pack_result(work / "result", out, name or f"часть-{part['index']}",
                          prepared["book"]["title"])
    print(f"\nготова часть {part['index']}: {archive.name} ({total / 60:.0f} мин звука)")
    return 0


def mode_final(parts: list, work, out, book, polish: str, name: str = "") -> int:
    """Финал: склейка частей, один прогон громкости и музыки на всю книгу, сборка."""
    banner(f"Финал: склеиваю {len(parts)} части в одну книгу")
    meta = book_meta(book)
    project = assemble_project(parts, meta)
    prepared_dir = work / "book"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    save(prepared_dir / "manifest.json", project)
    save(prepared_dir / "source.json", {"paragraphs": [{"id": pid, "text": text}
                                                       for pid, text in project["paragraph_texts"].items()]})
    highlight = "sentence"
    for part in parts:                                   # режим подсветки в частях один и тот же
        highlight = (load(part["source"]).get("highlight_mode") or highlight)
    banner("Манифест всей книги (та же подготовка, что на машине)")
    audiobook_epub.prepare(prepared_dir / "manifest.json", prepared_dir / "source",
                           highlight=highlight, start="p00001")
    whole = load(prepared_dir / "source" / "manifest.json")
    print(f"  книга: {whole['book']['title']!r} — глав {len(whole.get('chapters') or [])}, "
          f"сегментов {len(whole['segments'])}")

    banner("Стык: сходятся ли части с манифестом всей книги")
    rows = []
    for part in parts:
        rows.extend(load(part["source"])["segments"])
    count = verify_seam(whole, parts)
    spoken = 0
    for expected, actual in zip(whole["segments"], rows):
        if actual.get("tts_text"):
            spoken += 1
            expected["tts_text"] = actual["tts_text"]   # озвучен именно этот текст (ударения и числа)
    save(prepared_dir / "source" / "manifest.json", whole)
    print(f"  стык сошёлся: {count} сегментов, произносимый текст взят из частей у {spoken} из них")

    total = merge_audio(whole, parts, work)
    print(f"  звука в книге: {total / 3600:.2f} ч (сумма частей)")
    normalized, music = work / "audio-normalized", work / "audio-music"
    run_step("Громкость по всей книге (+ мягкая обработка речи)"
             if polish != "off" else "Громкость по всей книге",
             [sys.executable, REPO / "normalize_book_audio.py", "--audio-dir", work / "audio",
              "--out", normalized, "--polish", polish])
    run_step("Музыка: одна подложка на всю книгу",
             [sys.executable, REPO / "mix_book_music.py", "--audio-dir", normalized,
              "--out", music, "--volume-percent", "5"])
    run_step("Сборка m4b и read-along.epub",
             [sys.executable, REPO / "audiobook_epub.py", "package",
              "--manifest", prepared_dir / "source" / "manifest.json", "--audio-dir", music,
              "--out", work / "result"])
    base = name or (pathlib.Path(book).stem if book else "") or file_safe(whole["book"]["title"], "книга")
    archive = pack_result(work / "result", out, base, whole["book"]["title"])
    save(pathlib.Path(parts[0]["folder"]).parent / "glue.json",
         {"book": str(book), "title": whole["book"]["title"], "parts": len(parts),
          "segments": count, "seconds": total, "polish": polish, "archive": archive.name,
          "finished": __import__("time").time()})
    print(f"\nкнига собрана: {archive.name}  ({total / 3600:.2f} ч звука, частей {len(parts)})")
    print("сырьё частей осталось на месте: фильтр, громкость и музыку можно переделать и "
          "собрать заново, не трогая облако")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка книги из частей: m4b и read-along.epub")
    parser.add_argument("--mode", choices=["part", "final"], default="final",
                        help="part — одна часть как книга, final — вся книга из частей")
    parser.add_argument("--parts-dir", required=True, help="каталог с part-1, part-2, …")
    parser.add_argument("--part", type=int, default=0, help="номер части для --mode part")
    parser.add_argument("--out", required=True, help="куда положить готовый архив")
    parser.add_argument("--book", default="", help="исходный epub целиком (название, автор)")
    parser.add_argument("--polish", choices=["soft", "off"], default="soft",
                        help="мягкая обработка речи при выравнивании (off — только громкость)")
    parser.add_argument("--work", default="", help="рабочий каталог сборки (по умолчанию в --parts-dir)")
    parser.add_argument("--expect", type=int, default=0, help="сколько частей ждём")
    parser.add_argument("--name", default="", help="имя архива без пробелов (обычно имя файла книги)")
    args = parser.parse_args()

    banner("Параметры сборки")
    for key, value in vars(args).items():
        print(f"  {key:10} {value}")
    parts_dir = pathlib.Path(args.parts_dir)
    work = pathlib.Path(args.work) if args.work else parts_dir / "glue"
    work.mkdir(parents=True, exist_ok=True)
    if args.mode == "part":
        wanted = [item for item in check_parts(parts_dir, 0) if not args.part or item["index"] == args.part]
        if not wanted:
            raise SystemExit(f"части {args.part} нет в {parts_dir}")
        return mode_part(wanted, work, args.out, args.name)
    return mode_final(check_parts(parts_dir, args.expect), work, args.out, args.book, args.polish,
                      args.name)


if __name__ == "__main__":
    raise SystemExit(main())
