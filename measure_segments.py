#!/usr/bin/env python3
"""Диагностика нарезки: откуда берутся короткие сегменты.

Сегмент в этом конвейере — это кусок абзаца с одним голосом (реплика персонажа или речь
рассказчика). Реплики короче пары слов («— Да?») становятся отдельными сегментами, и модель
на них спотыкается. Здесь считаем по настоящим книгам, сколько таких сегментов и какие они.
"""
import pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prepare_qwen as pq
import russian_text as ru

SHORT = 60          # порог «слишком коротко»: примерно 3-4 секунды речи


def replica_split(paragraph):
    """Как конвейер видит абзац, если в нём диалог: реплики по тире — отдельные куски.

    Точную разбивку даёт LLM на стадии ролей, но короткие реплики видны и так: по тире.
    """
    text = paragraph["text"]
    if "\n" not in text and text.count("—") + text.count("–") == 0:
        return [text]
    parts = re.split(r"\s+[—–]\s*", text.strip())
    return [part for part in parts if part.strip()]


def lengths(path):
    source = pq.read_source(path)
    spans, paragraphs = [], 0
    for paragraph in source["paragraphs"]:
        paragraphs += 1
        for piece in replica_split(paragraph):
            for start, end, boundary in pq.split_text(piece):
                spans.append((piece[start:end].strip(), boundary))
    return source, paragraphs, spans


report = []
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.epub"))[:3]:
    source, paragraphs, spans = lengths(path)
    short = [(text, boundary) for text, boundary in spans if len(text) < SHORT]
    tiny = [(text, boundary) for text, boundary in spans if len(text) < 25]
    seconds = sum(len(text) for text, _ in spans) / 16.5
    report.append(f"\n=== {path.name}")
    report.append(f"абзацев {paragraphs}, сегментов {len(spans)}, звука ~{seconds / 60:.0f} мин")
    report.append(f"коротких (<{SHORT} симв.): {len(short)} ({len(short) * 100 / max(1, len(spans)):.1f}%), "
                  f"из них совсем крошки (<25): {len(tiny)}")
    sizes = sorted(len(text) for text, _ in spans)
    report.append(f"длины сегментов: медиана {sizes[len(sizes) // 2]}, "
                  f"самый короткий {sizes[0]}, самый длинный {sizes[-1]}")
    for text, boundary in sorted(tiny, key=len)[:12]:
        report.append(f"    «{text}»  [{boundary}]")
print("\n".join(report))
