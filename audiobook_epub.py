#!/usr/bin/env python3
"""EbookLib EPUB 3 + FFmpeg M4B, using the role manifest and measured timeline."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
from html import escape
import json
import math
from pathlib import Path
import posixpath
import re
import subprocess
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import uuid
from datetime import datetime, timezone

class _LazyEpub:
    """EbookLib подключается при первом обращении: на --help тяжёлые зависимости не нужны."""

    def __getattr__(self, name):
        import importlib
        return getattr(importlib.import_module("ebooklib.epub"), name)


epub = _LazyEpub()

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
SMIL = "http://www.w3.org/ns/SMIL"
XHTML = "http://www.w3.org/1999/xhtml"
EPUB = "http://www.idpf.org/2007/ops"
ET.register_namespace("", XHTML)
ET.register_namespace("epub", EPUB)
HIGHLIGHT_CSS = b"""body {line-height:1.5} p {white-space:pre-wrap}
span.tts-active, span.-epub-media-overlay-active {
 background-color:#ffe49a !important; color:#161616 !important;
 outline:1px solid #bb8500; border-radius:2px;
 -webkit-box-decoration-break:clone; box-decoration-break:clone;
}
.tts-playing {color:inherit}
"""


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def grouped(segments, key):
    result = OrderedDict()
    for segment in segments:
        result.setdefault(segment[key], []).append(segment)
    return result


def compact_anchored_text(path, manifest):
    """EbookLib pretty-prints inline spans; remove only those injected separators.

    All actual paragraph characters live inside our spans. Formatting whitespace
    outside them must not split words or introduce newlines under pre-wrap CSS.
    """
    documents = {f"EPUB/{cid}.xhtml" for cid in grouped(manifest["segments"], "chapter_id")}
    with tempfile.TemporaryDirectory(prefix="qwen-epub-text-") as folder:
        replacement = Path(folder) / "compact.epub"
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(replacement, "w") as target:
            for info in source.infolist():
                if info.filename in documents:
                    root = ET.fromstring(source.read(info.filename))
                    for paragraph in root.findall(".//{" + XHTML + "}p"):
                        paragraph.text = None
                        for child in paragraph:
                            child.tail = None
                    target.writestr(info, ET.tostring(root, encoding="utf-8", xml_declaration=True))
                else:
                    with source.open(info) as reader, target.open(info, "w") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        replacement.replace(path)


def select_excerpt(manifest, timeline, seconds, start):
    """Only whole paragraphs, using measured durations or a declared estimate."""
    segments = manifest["segments"]
    first = next((i for i, s in enumerate(segments) if s["paragraph_id"] == start), None)
    if first is None:
        raise ValueError(f"Нет абзаца {start}")
    measured = {s["segment_id"]: s for s in timeline}
    choices, selected, duration = [], [], 0.0
    for paragraph in grouped(segments[first:], "paragraph_id").values():
        if timeline and any(s["id"] not in measured for s in paragraph):
            break  # Never invent a measured duration for an incomplete paragraph.
        for s in paragraph:
            row = measured.get(s["id"])
            if row and row["text"] != s["text"]:
                raise ValueError("Измерения относятся к другому тексту")
            duration += (row["audio_end"] - row["audio_start"] if row else len(s["text"]) / 14) + s["pause_after_ms"] / 1000
        selected.extend(paragraph)
        choices.append((abs(duration - seconds), len(selected), duration))
        if duration >= seconds:
            break
    if not choices:
        raise ValueError("Нет полностью измеренного абзаца для фрагмента")
    _, count, duration = min(choices)
    return deepcopy(selected[:count]), duration


def select_range(manifest, timeline, start, end):
    """Inclusive whole-paragraph range; estimates only unmeasured segments."""
    paragraphs = list(grouped(manifest['segments'], 'paragraph_id'))
    if start not in paragraphs or end not in paragraphs:
        raise ValueError('Начальный или конечный абзац отсутствует')
    first, last = paragraphs.index(start), paragraphs.index(end)
    if first > last:
        raise ValueError('Конец фрагмента расположен раньше начала')
    included = set(paragraphs[first:last + 1])
    selected = [s for s in manifest['segments'] if s['paragraph_id'] in included]
    measured = {row['segment_id']: row for row in timeline}
    duration = 0.0
    for segment in selected:
        row = measured.get(segment['id'])
        if row and row['text'] != segment['text']:
            raise ValueError('Измерения относятся к другому тексту')
        duration += (row['audio_end'] - row['audio_start'] if row else len(segment['text']) / 14) + segment['pause_after_ms'] / 1000
    return deepcopy(selected), duration


def boundary_pause(segment, next_segment=None, sentence_ms=450, paragraph_ms=800):
    tail = re.sub(r'[\s»”"\'’\)\]\}—–-]+$', '', segment["text"])
    paragraph_end = next_segment is None or next_segment["paragraph_id"] != segment["paragraph_id"]
    if paragraph_end:
        return paragraph_ms
    if tail.endswith(("…", "...")):
        return max(sentence_ms, 700)
    if tail.endswith(("?", "!")):
        return max(sentence_ms, 600)
    if tail.endswith("."):
        return sentence_ms
    return 180 if next_segment["speaker"] != segment["speaker"] else 100


def source_pieces(segment):
    """Куски абзацев, из которых собран сегмент (склейка коротких соседей даёт несколько)."""
    if segment.get("sources"):
        return segment["sources"]
    return [{"paragraph_id": segment["paragraph_id"], "text": segment["text"],
             "char_start": segment["source"]["char_start"], "char_end": segment["source"]["char_end"]}]


def sentence_segments(manifest, sentence_ms=450, paragraph_ms=800):
    from razdel import sentenize
    from types import SimpleNamespace
    result = []
    for parent in manifest["segments"]:
        number = 0
        for piece in source_pieces(parent):
            sentences = list(sentenize(piece["text"]))
            # Razdel may emit the closing dialogue dash as its own sentence.
            # Attach it to the preceding sentence within the SAME role/source span.
            joined = []
            for sentence in sentences:
                if joined and not any(c.isalnum() for c in sentence.text):
                    previous = joined[-1]
                    joined[-1] = SimpleNamespace(start=previous.start, stop=sentence.stop,
                        text=piece["text"][previous.start:sentence.stop])
                else:
                    joined.append(sentence)
            sentences = joined
            if not sentences:
                raise ValueError("Пустой фрагмент: " + parent["id"])
            base = piece["char_start"]
            paragraph = manifest.get("paragraph_texts", {}).get(piece["paragraph_id"])
            if paragraph is not None:
                raw = paragraph[base:piece["char_end"]]
                if raw.strip() != piece["text"]:
                    raise ValueError("Источник не соответствует тексту для деления на предложения")
                base += len(raw) - len(raw.lstrip())
            for sentence in sentences:
                number += 1
                row = deepcopy(parent)
                # TTS offsets differ after numeral expansion; never slice/inherit a
                # parent's normalized speech target into a newly split sentence.
                for key in list(row):
                    if key.startswith("tts_"):
                        del row[key]
                row.pop("sources", None)
                # Склеенный из нескольких абзацев сегмент всегда делится: иначе первый кусок
                # занял бы идентификатор всего сегмента и разошёлся бы с дорожкой синтеза.
                suffixed = len(sentences) > 1 or len(source_pieces(parent)) > 1
                row["id"] = parent["id"] + f"_s{number:03d}" if suffixed else parent["id"]
                row["sentence_parent"] = parent["id"]
                row["synthesis_group"] = parent["id"]
                row["paragraph_id"] = piece["paragraph_id"]
                row["text"] = sentence.text
                row["source"] = dict(parent["source"], char_start=base + sentence.start,
                                     char_end=base + sentence.stop)
                row["pause_after_ms"] = parent["pause_after_ms"]
                row["audio_file"] = "audio/" + row["id"] + ".wav"
                row["split_boundary"] = "sentence"
                row["order"] = len(result)
                result.append(row)
    for i, row in enumerate(result):
        row["pause_after_ms"] = boundary_pause(row, result[i + 1] if i + 1 < len(result) else None, sentence_ms, paragraph_ms)
    return seal_groups(result)


def piece_segments(manifest):
    """Режим fragment: склеенный из нескольких абзацев сегмент → по строке на абзац.

    Подсветка живёт внутри абзаца (один span — один <p>), поэтому склейку коротких абзацев
    нельзя показать одним span. Строки остаются одной группой синтеза: модель всё равно
    получает их одним вызовом, а звук делится между строками уже после синтеза.
    """
    result = []
    for parent in manifest["segments"]:
        pieces = source_pieces(parent)
        if len(pieces) == 1:
            row = deepcopy(parent)
            row["synthesis_group"] = parent["id"]
            result.append(row)
            continue
        for number, piece in enumerate(pieces, 1):
            row = deepcopy(parent)
            for key in list(row):
                if key.startswith("tts_"):
                    del row[key]
            row.pop("sources", None)
            row["id"] = parent["id"] + f"_p{number:03d}"
            row["synthesis_group"] = parent["id"]
            row["paragraph_id"] = piece["paragraph_id"]
            row["text"] = piece["text"]
            row["source"] = dict(parent["source"], char_start=piece["char_start"], char_end=piece["char_end"])
            row["audio_file"] = "audio/" + row["id"] + ".wav"
            result.append(row)
    return seal_groups(result)


def seal_groups(rows):
    """Внутри группы синтеза паузу между строками даёт сама модель: монтажную не добавляем.

    Группа — это один вызов модели (склейка до ~220 знаков); её звук режется между строками
    по естественным паузам, и склеенные обратно куски должны дать ровно исходный звук.
    """
    for row, following in zip(rows, rows[1:]):
        if row.get("synthesis_group") is not None and row.get("synthesis_group") == following.get("synthesis_group"):
            row["pause_after_ms"] = 0
    return rows


def build_text_epub(manifest, destination, cover_dir=None):
    """Rebuild normalized text, paragraph boundaries and segment anchors with EbookLib."""
    book = epub.EpubBook()
    book.set_identifier("urn:sha256:" + hashlib.sha256(json.dumps(manifest["segments"], ensure_ascii=False).encode()).hexdigest())
    book.set_title(manifest["book"]["title"])
    book.set_language("ru")
    book.add_author(manifest["book"].get("author", ""))
    cover = manifest["book"].get("cover")
    if cover and cover_dir is not None:
        # Только картинка с cover-image, без отдельной страницы: страница обложки не текст
        # книги и не должна попадать в spine и в синхронизацию.
        book.set_cover("images/" + cover["file"], (Path(cover_dir) / cover["file"]).read_bytes(), create_page=False)
    book.add_item(epub.EpubItem(uid="style", file_name="style.css", media_type="text/css",
        content=HIGHLIGHT_CSS))
    docs = []
    for chapter, rows in grouped(manifest["segments"], "chapter_id").items():
        content = []
        for index, (pid, spans) in enumerate(grouped(rows, "paragraph_id").items()):
            cursor = 0
            parts = []
            paragraph_text = manifest.get("paragraph_texts", {}).get(pid)
            def gap(text):
                if text:
                    parts.append('<span class="text-gap">' + escape(text) + '</span>')
            for span in spans:
                original = deepcopy(span["source"])
                span.setdefault("original_source", original)
                start, end = cursor, cursor + len(span["text"])
                if paragraph_text is not None:
                    raw_start, raw_end = original["char_start"], original["char_end"]
                    raw = paragraph_text[raw_start:raw_end]
                    if raw.strip() != span["text"] or raw_start < cursor:
                        raise ValueError("Фрагмент не соответствует исходному абзацу: " + span["id"])
                    start = raw_start + len(raw) - len(raw.lstrip())
                    end = start + len(span["text"])
                    gap(paragraph_text[cursor:start])
                parts.append(f'<span id="{escape(span["id"])}" data-speaker="{escape(span["speaker"])}">{escape(span["text"])}</span>')
                span["source"] = {"href": f"EPUB/{chapter}.xhtml", "block_index": index,
                    "element_id": pid, "tag": "p", "char_start": start, "char_end": end}
                cursor = end
            if paragraph_text is not None:
                gap(paragraph_text[cursor:])
            content.append(f'<p id="{escape(pid)}">' + "".join(parts) + "</p>")
        chapter_title = next((c["title"] for c in manifest.get("chapters", []) if c["id"] == chapter), manifest["book"]["title"])
        doc = epub.EpubHtml(uid=chapter, file_name=f"{chapter}.xhtml", title=chapter_title, lang="ru")
        doc.content = "<html><body>" + "\n".join(content) + "</body></html>"
        doc.add_link(href="style.css", rel="stylesheet", type="text/css")
        book.add_item(doc)
        docs.append(doc)
    book.toc = tuple(docs)
    book.spine = docs  # Navigation is not spoken content.
    book.add_item(epub.EpubNav(title="Оглавление"))
    book.add_item(epub.EpubNcx())
    epub.write_epub(str(destination), book, {"raise_exceptions": True, "epub3_pages": False})
    compact_anchored_text(destination, manifest)
    manifest["book"]["input_sha256"] = hashlib.sha256(Path(destination).read_bytes()).hexdigest()
    manifest["book"]["epub_path"] = str(Path(destination).resolve())
    validate_epub(destination, manifest)


def prepare(manifest_path, out, seconds=None, timing=None, start="p00058", highlight="sentence", sentence_ms=450, paragraph_ms=800, end=None):
    manifest = load(manifest_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    original_book = deepcopy(manifest["book"])
    duration = None
    if end is not None:
        manifest['segments'], duration = select_range(manifest, load(timing)['segments'] if timing else [], start, end)
        manifest['book']['title'] += ' — тестовый фрагмент'
    elif seconds is not None:
        manifest["segments"], duration = select_excerpt(manifest, load(timing)["segments"] if timing else [], seconds, start)
        manifest["book"]["title"] += " — тестовый фрагмент"
    used = {s["speaker"] for s in manifest["segments"]}
    source_json = Path(manifest_path).parent / "source.json"
    if source_json.exists():
        selected_paragraphs = {piece["paragraph_id"] for s in manifest["segments"]
                               for piece in source_pieces(s)}
        manifest["paragraph_texts"] = {p["id"]: p["text"] for p in load(source_json)["paragraphs"] if p["id"] in selected_paragraphs}
    if highlight == "sentence":
        manifest["segments"] = sentence_segments(manifest, sentence_ms, paragraph_ms)
    else:
        manifest["segments"] = piece_segments(manifest)
    manifest.pop("speech_preparation", None)
    manifest["highlight_mode"] = highlight
    manifest["pause_policy"] = {"sentence_ms": sentence_ms, "paragraph_ms": paragraph_ms,
                                "question_exclamation_ms": max(sentence_ms, 600), "ellipsis_ms": max(sentence_ms, 700),
                                "mid_sentence_role_ms": 180}
    manifest["voices"] = [v for v in manifest["voices"] if v["id"] in used]
    for i, segment in enumerate(manifest["segments"]):
        segment["order"] = i
    manifest["origin_book"] = original_book
    cover = manifest["book"].get("cover")
    if cover:
        # Обложка едет вместе с подготовкой: из неё её берут и текстовый EPUB, и M4B.
        found = Path(manifest_path).parent / cover["file"]
        if found.is_file() and hashlib.sha256(found.read_bytes()).hexdigest() == cover["sha256"]:
            if found.resolve() != (out / cover["file"]).resolve():
                shutil.copyfile(found, out / cover["file"])
        else:
            print("Обложка не найдена рядом с манифестом — книга соберётся без неё:", found, flush=True)
            manifest["book"].pop("cover")
    if not manifest.get("segments"):
        raise ValueError("Манифест без сегментов: нечего готовить и упаковывать")
    manifest["excerpt"] = {"target_seconds": seconds, "expected_seconds": duration,
        "duration_basis": "measured where available; remainder estimated at 14 characters per second" if end else ("measured first test" if timing else "estimated 14 characters per second"),
        "first_paragraph": manifest["segments"][0]["paragraph_id"], "last_paragraph": manifest["segments"][-1]["paragraph_id"]}
    text_epub = out / "book-text.epub"
    build_text_epub(manifest, text_epub, cover_dir=out)
    manifest["summary"] = {"segments": len(manifest["segments"]), "paragraphs": len(grouped(manifest["segments"], "paragraph_id")),
        "voices": len(manifest["voices"])}
    save(out / "manifest.json", manifest)
    print(json.dumps(dict(manifest["summary"], **manifest["excerpt"], epub=str(text_epub)), ensure_ascii=False))


def verify_timeline(manifest, timeline, total):
    if [s["id"] for s in manifest["segments"]] != [s["segment_id"] for s in timeline]:
        raise ValueError("Не озвучена вся книга или изменён порядок фрагментов")
    previous = 0.0
    for segment, row in zip(manifest["segments"], timeline):
        begin, end = row["audio_start"], row["audio_end"]
        if not all(math.isfinite(n) for n in (begin, end)) or not 0 <= previous <= begin < end <= total:
            raise ValueError("Некорректные временные привязки")
        if (segment["text"], segment["speaker"], segment["source"]) != (row["text"], row["speaker"], row["source"]):
            raise ValueError("Текст, роль или источник аудио не соответствует EPUB")
        if segment.get("tts_text") != row.get("tts_text"):
            raise ValueError("Нормализованный текст аудио не соответствует манифесту")
        previous = end


def metadata_value(value):
    return str(value).replace("\\", "\\\\").replace("\n", " ").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")


def package(manifest_path, audio_dir, out):
    candidate = load(manifest_path)
    if candidate.get("speech_preparation"):
        from prepare_speech import check
        check(candidate)
    manifest, audio_dir, out = load(manifest_path), Path(audio_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    timeline = load(audio_dir / "timeline.json")["segments"]
    report = load(audio_dir / "benchmark.json")
    total = report["final_audio_seconds"]
    if report.get("state") != "complete":
        raise ValueError("Аудиопрогон не завершён")
    if report.get("book", {}).get("input_sha256") != manifest["book"]["input_sha256"]:
        raise ValueError("Аудиопрогон относится к другому EPUB")
    verify_timeline(manifest, timeline, total)
    source_epub = Path(manifest["book"]["epub_path"])
    if hashlib.sha256(source_epub.read_bytes()).hexdigest() != manifest["book"]["input_sha256"]:
        raise ValueError("Текстовый EPUB изменён после подготовки")
    actual = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_format", "-of", "json", str(audio_dir / "sample.wav")]))
    actual_duration = (actual.get("format") or {}).get("duration")
    if actual_duration is None:
        raise ValueError("ffprobe не вернул длительность sample.wav — файл повреждён")
    if abs(float(actual_duration) - total) > 0.001:
        raise ValueError("WAV и измеренная длительность не соответствуют друг другу")
    chapters = []
    for chapter_id, rows in grouped(manifest["segments"], "chapter_id").items():
        start_index = next(i for i, row in enumerate(timeline) if row["segment_id"] == rows[0]["id"])
        chapters.append((chapter_id, timeline[start_index]["audio_start"]))
    metadata = [";FFMETADATA1", "title=" + metadata_value(manifest["book"]["title"]),
        "artist=" + metadata_value(manifest["book"].get("author", "")), "genre=Audiobook"]
    for i, (cid, begin) in enumerate(chapters):
        end = chapters[i + 1][1] if i + 1 < len(chapters) else total
        title = next((c["title"] for c in manifest.get("chapters", []) if c["id"] == cid), manifest["book"]["title"])
        metadata.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(begin * 1000)}", f"END={round(end * 1000)}", "title=" + metadata_value(title)])
    with tempfile.TemporaryDirectory(prefix="qwen-book-package-") as folder:
        meta = Path(folder) / "chapters.txt"
        meta.write_text("\n".join(metadata) + "\n", encoding="utf-8")
        m4b = out / "book.m4b"
        temp_m4b = Path(folder) / "book.m4b"
        cover = manifest["book"].get("cover")
        cover_file = Path(manifest_path).parent / cover["file"] if cover else None
        if cover_file is not None and not cover_file.is_file():
            raise ValueError("Обложка из манифеста пропала: " + str(cover_file))
        artwork = []
        if cover_file is not None:
            # Обложка — attached_pic: её показывают плееры аудиокниг. JPEG и PNG MP4 хранит как
            # есть, остальные форматы перекодируем в JPEG.
            codec = "copy" if cover["media_type"] in {"image/jpeg", "image/png"} else "mjpeg"
            artwork = ["-i", str(cover_file), "-map", "2:v:0", "-c:v", codec, "-disposition:v:0", "attached_pic"]
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_dir / "sample.wav"),
            "-f", "ffmetadata", "-i", str(meta), *artwork[:2], "-map", "0:a:0", *artwork[2:],
            "-map_metadata", "1", "-map_chapters", "1",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-f", "mp4", str(temp_m4b)], check=True)
        probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-show_chapters", "-of", "json", str(temp_m4b)]))
        if cover_file is not None and not any(s.get("disposition", {}).get("attached_pic") for s in probe["streams"]):
            raise ValueError("Обложка не попала в M4B")
        probe_duration = (probe.get("format") or {}).get("duration")
        if probe_duration is None:
            raise ValueError("ffprobe не вернул длительность собранного M4B — файл повреждён")
        if abs(float(probe_duration) - total) > 0.15 or len(probe["chapters"]) != len(chapters):
            raise ValueError("Длительность или главы M4B не соответствуют книге")
        if not any(s.get("codec_name") == "aac" for s in probe["streams"]):
            raise ValueError("M4B не содержит AAC")
        # Use EbookLib for reading/rewriting OPF, navigation and SMIL item declarations.
        book = epub.read_epub(str(source_epub), {"ignore_ncx": False})
        book.add_prefix("media", "http://www.idpf.org/epub/vocab/overlays/#")
        book.add_metadata(None, "meta", "tts-active", {"property": "media:active-class"})
        book.add_metadata(None, "meta", "tts-playing", {"property": "media:playback-active-class"})
        overlay_duration = 0.0
        for cid, rows in grouped(manifest["segments"], "chapter_id").items():
            root = ET.Element("smil", {"xmlns": SMIL, "xmlns:epub": EPUB, "version": "3.0"})
            seq = ET.SubElement(ET.SubElement(root, "body"), "seq", {"id": "seq-" + cid, "epub:textref": cid + ".xhtml"})
            chapter_duration = 0.0
            for segment in rows:
                i = segment["order"]
                row = timeline[i]
                clip_end = timeline[i + 1]["audio_start"] if i + 1 < len(timeline) else total
                par = ET.SubElement(seq, "par", {"id": "par-" + segment["id"]})
                ET.SubElement(par, "text", {"src": f"{cid}.xhtml#{segment['id']}"})
                ET.SubElement(par, "audio", {"src": "audio/book.m4a", "clipBegin": f"{row['audio_start']:.6f}s", "clipEnd": f"{clip_end:.6f}s"})
                chapter_duration += clip_end - row["audio_start"]
            overlay_duration += chapter_duration
            overlay_id = "overlay-" + cid
            doc = book.get_item_with_id(cid)
            doc.media_overlay = overlay_id
            # EbookLib's reader does not restore head links/lang/title on EpubHtml.
            doc.lang = "ru"
            doc.title = manifest["book"]["title"]
            doc.add_link(href="style.css", rel="stylesheet", type="text/css")
            book.add_item(epub.EpubItem(uid=overlay_id, file_name=cid + ".smil", media_type="application/smil+xml", content=ET.tostring(root, encoding="utf-8", xml_declaration=True)))
            book.add_metadata(None, "meta", f"{chapter_duration:.6f}s", {"property": "media:duration", "refines": "#" + overlay_id})
        book.add_metadata(None, "meta", f"{overlay_duration:.6f}s", {"property": "media:duration"})
        # Declare the audio through EbookLib, stream its bytes into the ZIP afterward.
        book.add_item(epub.EpubItem(uid="book-audio", file_name="audio/book.m4a", media_type="audio/mp4", content=b""))
        intermediate = Path(folder) / "overlay.epub"
        epub.write_epub(str(intermediate), book, {"raise_exceptions": True, "epub3_pages": False})
        compact_anchored_text(intermediate, manifest)
        final_epub = out / "book-read-along.epub"
        temp_epub = Path(folder) / "final.epub"
        with zipfile.ZipFile(intermediate) as source, zipfile.ZipFile(temp_epub, "w") as target:
            for entry in source.infolist():
                if entry.filename != "EPUB/audio/book.m4a":
                    target.writestr(entry, source.read(entry.filename))
            target.write(temp_m4b, "EPUB/audio/book.m4a", compress_type=zipfile.ZIP_STORED)
        validation = validate_epub(temp_epub, manifest, overlays=True)
        temp_m4b.replace(m4b)
        temp_epub.replace(final_epub)
    save(out / "package-report.json", {"state": "complete", "audio_seconds": total, "chapters": len(chapters),
        "epub": str(final_epub.resolve()), "m4b": str(m4b.resolve()), "validation": validation,
        "sync_granularity": manifest.get("highlight_mode", "fragment"), "builder": "EbookLib 0.20 + FFmpeg"})
    print(f"Ready: {m4b}\nRead along: {final_epub}")


def validate_epub(path, manifest, overlays=False):
    count = 0
    with zipfile.ZipFile(path) as archive:
        if archive.infolist()[0].filename != "mimetype" or archive.infolist()[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError("EPUB mimetype must be first and uncompressed")
        if archive.read("mimetype") != b"application/epub+zip" or archive.testzip():
            raise ValueError("Повреждён ZIP EPUB")
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        opf_name = next(e.attrib["full-path"] for e in container.iter() if e.tag.endswith("}rootfile"))
        opf = ET.fromstring(archive.read(opf_name))
        items = {e.get("id"): e for e in opf.iter() if e.tag.endswith("}item")}
        for item in items.values():
            name = posixpath.join(posixpath.dirname(opf_name), item.get("href"))
            if name not in archive.namelist():
                raise ValueError("EPUB resource missing: " + name)
        for cid, rows in grouped(manifest["segments"], "chapter_id").items():
            doc_name = posixpath.join(posixpath.dirname(opf_name), items[cid].get("href"))
            doc = ET.fromstring(archive.read(doc_name))
            anchors = {e.get("id"): e for e in doc.iter() if e.get("id")}
            for pid, text in manifest.get("paragraph_texts", {}).items():
                if pid in anchors and "".join(anchors[pid].itertext()) != text:
                    raise ValueError("Текст абзаца EPUB изменился: " + pid)
            for s in rows:
                if s["id"] not in anchors or "".join(anchors[s["id"]].itertext()) != s["text"]:
                    raise ValueError("EPUB text or anchor mismatch: " + s["id"])
            if overlays:
                links = doc.findall(".//{" + XHTML + "}link")
                if not any(link.get("href") == "style.css" and link.get("rel") == "stylesheet" for link in links):
                    raise ValueError("В EPUB не подключён стиль подсветки")
                overlay = items[items[cid].get("media-overlay")]
                root = ET.fromstring(archive.read(posixpath.join(posixpath.dirname(opf_name), overlay.get("href"))))
                pars = root.findall(".//{" + SMIL + "}par")
                if len(pars) != len(rows):
                    raise ValueError("Неполная синхронизация EPUB")
                for par, s in zip(pars, rows):
                    if par.find("{" + SMIL + "}text").get("src") != f"{cid}.xhtml#{s['id']}":
                        raise ValueError("Неверная ссылка SMIL")
                count += len(pars)
    return {"text_segments": len(manifest["segments"]), "synchronized_segments": count, "checks": "ZIP CRC, resources, exact text, anchors, overlay coverage"}


def repair_highlighting(source, destination):
    """Repair an existing audio EPUB without touching encoded audio or timing."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen-highlight-") as folder:
        repaired = Path(folder) / "highlight.epub"
        with zipfile.ZipFile(source) as reader, zipfile.ZipFile(repaired, "w") as writer:
            for entry in reader.infolist():
                if entry.filename == "EPUB/style.css":
                    writer.writestr(entry, HIGHLIGHT_CSS)
                elif entry.filename.endswith(".xhtml") and entry.filename != "EPUB/nav.xhtml":
                    root = ET.fromstring(reader.read(entry.filename))
                    root.set("lang", "ru")
                    root.set("{http://www.w3.org/XML/1998/namespace}lang", "ru")
                    head = root.find("{" + XHTML + "}head")
                    if head is None:
                        head = ET.Element("{" + XHTML + "}head")
                        root.insert(0, head)
                    if head.find("{" + XHTML + "}title") is None:
                        ET.SubElement(head, "{" + XHTML + "}title").text = "Скуки ради — подсветка"
                    if not any(e.get("href") == "style.css" for e in head):
                        ET.SubElement(head, "{" + XHTML + "}link", {"rel": "stylesheet", "type": "text/css", "href": "style.css"})
                    writer.writestr(entry, ET.tostring(root, encoding="utf-8", xml_declaration=True))
                elif entry.filename == "EPUB/content.opf":
                    root = ET.fromstring(reader.read(entry.filename))
                    for element in root.iter():
                        if element.tag.endswith("}identifier") and element.get("id") == root.get("unique-identifier"):
                            element.text = "urn:uuid:" + str(uuid.uuid4())
                        elif element.tag.endswith("}title"):
                            element.text = (element.text or "Книга") + " — подсветка"
                        elif element.get("property") == "dcterms:modified":
                            element.text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    writer.writestr(entry, ET.tostring(root, encoding="utf-8", xml_declaration=True))
                else:
                    with reader.open(entry) as input_stream, writer.open(entry, "w") as output_stream:
                        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        repaired.replace(destination)
    print(f"Highlight styles repaired; original audio preserved: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--seconds", type=float)
    prep.add_argument("--timing")
    prep.add_argument("--start-paragraph", default="p00058")
    prep.add_argument("--end-paragraph", help="Включительный конец фиксированного фрагмента")
    prep.add_argument("--highlight", choices=["sentence", "fragment"], default="sentence")
    prep.add_argument("--sentence-pause-ms", type=int, default=450)
    prep.add_argument("--paragraph-pause-ms", type=int, default=800)
    pack = commands.add_parser("package")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--audio-dir", required=True)
    pack.add_argument("--out", required=True)
    repair = commands.add_parser("repair-highlight")
    repair.add_argument("--epub", required=True)
    repair.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.seconds is not None and args.seconds <= 0:
                raise ValueError("Длительность должна быть положительной")
            if not 100 <= args.sentence_pause_ms <= 2000 or not args.sentence_pause_ms <= args.paragraph_pause_ms <= 4000:
                raise ValueError("Некорректные паузы")
            prepare(args.manifest, args.out, args.seconds, args.timing, args.start_paragraph, args.highlight,
                    args.sentence_pause_ms, args.paragraph_pause_ms, args.end_paragraph)
        elif args.command == "package":
            package(args.manifest, args.audio_dir, args.out)
        else:
            repair_highlighting(args.epub, args.out)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
