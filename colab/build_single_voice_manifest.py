#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPUB → манифест проекта БЕЗ облачной LLM (один чтец на всю книгу).

Использует собственные функции репозитория (prepare_qwen.py: read_source, validate_cast,
validate_roles, export_project), поэтому все проверки целостности текста остаются штатными.

Репозиторий запрещает отдавать прямую речь рассказчику, поэтому каждый абзац режется
по репликам: «— ...» — голос dialog, слова автора — narrator. Оба голоса используют
ОДИН И ТОТ ЖЕ образец (вшитый WAV), так что книга читается одним тембром.
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
# Скрипт лежит в colab/, а модули репозитория — в корне: находим корень в любом случае.
REPO = HERE.parent if (HERE.parent / "prepare_qwen.py").is_file() else HERE
sys.path.insert(0, str(REPO))
from prepare_qwen import export_project, read_source, validate_cast, validate_roles, write  # noqa: E402

DASH = re.compile(r"(?:^|\s)[—–]\s+")          # тире реплики: в начале абзаца или после пробела
REPORTING = re.compile(                               # те же авторские ремарки, что и в prepare_qwen
    r"(?:сказал|сказала|ответил|ответила|проговорил|произнёс|произнес|промолвил|спросил|спросила|"
    r"воскликнул|заметил|продолжил|продолжал|прибавил|повторил|перебил|подхватил|возразил|"
    r"обратился|прошептал|закричал|пробормотал|вздохнул|усмехнулся|улыбнулся|кивнул|"
    r"говорит|говорил|говорила|читал|читала|подумал|вспомнил|решил|молвил|начал|кончил)", re.I)

NARRATOR_DESCRIPTION = (
    "Adult Russian male, about 35 to 50. Warm mature baritone with a full, resonant chest tone. "
    "Smooth, clear timbre, no rasp. Calm, attentive and restrained. Moderate pace, even breathing, "
    "clear diction. Natural sense accents, small pauses between thoughts. Steady, soft phrase endings. "
    "Speak Russian clearly. Keep one consistent voice. Say only the supplied text. "
    "No extra utterances or background sounds."
)
DIALOG_DESCRIPTION = (
    "Same reader as the narrator, deliberately unchanged: this production uses one voice for the whole book, "
    "speaking dialogue lines with the same timbre, pitch and pace as the narrative. "
    "Natural Russian speech, clear diction, no accent, no theatrical overacting, no character acting."
)


def split_paragraph(text, dialogue_id="dialog"):
    """Абзац → сегменты: реплики («— …») уходят голосу dialogue_id, слова автора — narrator."""
    cuts = [m.start() for m in DASH.finditer(text)]
    pieces, previous = [], 0
    for cut in cuts:
        if cut > previous:
            pieces.append(text[previous:cut])
        previous = cut
    if previous < len(text):
        pieces.append(text[previous:])
    spans, cursor = [], 0
    for piece in pieces:
        value = piece.strip()
        if not value:
            continue
        start = text.index(value, cursor)
        cursor = start + len(value)
        speaker = dialogue_id if value.startswith(("–", "—")) else "narrator"
        spans.append({"start": start, "end": cursor, "speaker": speaker,
                      "confidence": "high", "reason": ""})
    if not spans:
        spans = [{"start": 0, "end": len(text), "speaker": "narrator", "confidence": "high", "reason": ""}]
    return spans


def build(book, out, target_chars=220, max_chars=260, min_chars=100, dialogue_id="dialog"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = read_source(str(book))
    write(out / "source.json", source)
    write(out / "status.json", {"state": "preparing", "input_sha256": source["input_sha256"]})

    cast = validate_cast({"characters": [
        dict(id="narrator", name="Рассказчик", gender="male", age="adult", aliases=[],
             evidence="Слова автора и повествование (облачная LLM не использовалась).",
             voice_description=NARRATOR_DESCRIPTION),
        dict(id=dialogue_id, name="Реплики (тот же чтец)", gender="male", age="adult", aliases=[],
             evidence="Прямая речь; читает тот же чтец, образец голоса тот же (облачная LLM не использовалась).",
             voice_description=DIALOG_DESCRIPTION)]})

    paragraphs = source["paragraphs"]
    labels, dialogue_spans = [], 0
    for i in range(0, len(paragraphs), 200):
        batch = paragraphs[i:i + 200]
        answers = [{"id": p["id"], "segments": [{"text": p["text"][s["start"]:s["end"]],
                                                "speaker": s["speaker"], "confidence": s["confidence"],
                                                "reason": s["reason"]}
                                               for s in split_paragraph(p["text"], dialogue_id)]}
                   for p in batch]
        labels.extend(validate_roles({"paragraphs": answers}, batch, cast))
    dialogue_spans = sum(1 for l in labels for s in l["spans"] if s["speaker"] == dialogue_id)

    write(out / "cast.json", dict(cast, input_sha256=source["input_sha256"]))
    write(out / "roles.json", {"paragraphs": labels})

    args = SimpleNamespace(target_chars=target_chars, max_chars=max_chars, min_chars=min_chars,
                           model="one-reader",
                           no_think=True, roles_overrides=None)
    export_project(out, source, cast, labels, args)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"manifest": str(out / "manifest.json"), "paragraphs": len(paragraphs),
                      "segments": len(manifest["segments"]), "dialogue_segments": dialogue_spans,
                      "voices": len(manifest["voices"]), "chapters": len(manifest["chapters"]),
                      "review": manifest["stats"]["review_segments"]}, ensure_ascii=False))
    return out / "manifest.json"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EPUB → манифест проекта с одним чтецом (без LLM)")
    parser.add_argument("book")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--target-chars", type=int, default=220)
    parser.add_argument("--max-chars", type=int, default=260)
    parser.add_argument("--min-chars", type=int, default=100)
    a = parser.parse_args()
    build(a.book, a.out, a.target_chars, a.max_chars, a.min_chars)
