#!/usr/bin/env python3
"""Post-role speech preparation: Supertonic normalization -> contextual LLM stress.

Always run AFTER final role/sentence segmentation and BEFORE VoiceDesign/Base.
Only tts_* fields are added; text/source/role/EPUB anchors remain untouched.
"""
import argparse
from copy import deepcopy
from difflib import SequenceMatcher
import getpass
import json
import os
from pathlib import Path
import re

from prepare_qwen import digest, load, obtain, write
import russian_text as ru

VERSION = "speech-preparation-1"
WORD = re.compile(r"\w+(?:\u0301\w*)*", re.UNICODE)
# БЕЗ (БЕЗО) normally has no independent stress (Gramota's pronunciation
# dictionary, https://gramota.ru/meta/bez). Do not force a fabricated acute.
# Keep this deliberately narrow: ambiguous 'надо' and ordinary words must
# still pass the normal stress-coverage check.
UNSTRESSED_PREPOSITIONS = frozenset({"безо"})


def needs_stress(word, text="", start=0):
    if word.lower() in UNSTRESSED_PREPOSITIONS:
        return False
    # Enclitic parts of кто-нибудь / что-либо share the word's stress; they
    # are not independently stressed words. Standalone 'либо' is NOT exempt.
    if word.lower() in {"нибудь", "либо"} and start > 0 and text[start-1] == "-":
        return False
    return (bool(re.fullmatch(r"[А-Яа-яЁё\u0301]+", word)) and
            sum(c in ru.VOWELS for c in word.lower()) > 1 and
            "ё" not in word.lower() and "\u0301" not in word)


def bare(word):
    return word.replace("\u0301", "").replace("ё", "е").replace("Ё", "Е")


def numeral_family(word):
    key = bare(word.lower())
    for nominative, forms in ru.FORMS.items():
        if key in {bare(f) for f in forms}:
            # one/two can change gender, but not value; scales retain the scale.
            nominative = {"одна": "один", "одно": "один", "две": "два"}.get(nominative, nominative)
            values = dict(zip(ru.UNITS, range(10)))
            values.update(zip(ru.TEENS, range(10, 20)))
            values.update({w: i * 10 for i, w in enumerate(ru.TENS) if i >= 2})
            values.update({w: i * 100 for i, w in enumerate(ru.HUNDREDS) if i >= 1})
            for stem, value in [("тысяч", 1000), ("миллион", 10**6), ("миллиард", 10**9)]:
                if nominative.startswith(stem):
                    return value
            return values[nominative]
    endings = "ьими|ьего|ьему|ыми|ьей|ьим|ьих|ьем|ого|ому|ый|ой|ая|ую|ое|ые|ых|ым|ом|ий|ья|ье|ью"
    for n, root in ru.ROOTS.items():
        root = "трет" if root == "треть" else bare(root)
        if re.fullmatch(re.escape(root) + "(?:" + endings + ")", key):
            return n
    return None


def numeral_positions(source, normalized):
    """Only freshly expanded numbers may be inflected, not author's written words."""
    original = [bare(w) for w in WORD.findall(source)]
    target = WORD.findall(normalized)
    allowed = set()
    for op, _, _, start, stop in SequenceMatcher(None, original, [bare(w) for w in target], autojunk=False).get_opcodes():
        if op in {"replace", "insert"}:
            allowed.update(i for i in range(start, stop) if numeral_family(target[i]) is not None)
    return allowed


def validate_text(original, prepared, stress=False, allowed_numerals=None):
    if not isinstance(prepared, str) or not prepared.strip():
        raise ValueError("Пустой/нестроковый текст")
    before, after = WORD.findall(original), WORD.findall(prepared)
    matches = list(WORD.finditer(prepared))
    opaque = r"(?:https?://|www\.)[^\s<>]+|[\w.+-]+@[\w.-]+\.[A-Za-z]+"
    if re.findall(opaque, original) != re.findall(opaque, prepared):
        raise ValueError("Изменена ссылка или адрес электронной почты")
    if len(before) != len(after):
        raise ValueError("Изменилось число слов: запрещено добавлять/удалять текст")
    for index, (a, b) in enumerate(zip(before, after)):
        if bare(a) != bare(b):
            fa, fb = numeral_family(a), numeral_family(b)
            if fa is None or fa != fb or a[0].isupper() != b[0].isupper() or allowed_numerals is not None and index not in allowed_numerals:
                raise ValueError("LLM изменила слово или значение числа: " + a)
        if "ё" in a.lower() and "ё" not in b.lower():
            raise ValueError("Удалена исходная ё")
        if "\u0301" in a and a != b:
            raise ValueError("Изменено исходное ударение")
        if b.count("\u0301") > 1 or any(i == 0 or b[i-1].lower() not in ru.VOWELS for i, c in enumerate(b) if c == "\u0301"):
            raise ValueError("Некорректное ударение")
        if stress and needs_stress(b, prepared, matches[index].start()):
            raise ValueError("Пропущено ударение: " + b)
    # Permit only the punctuation work requested by Supertonic. Keep quotes,
    # brackets, line boundaries and word boundaries, including hyphenated words.
    def skeleton(text):
        return re.sub(r"[,.!?;:—–…]", "", WORD.sub("W", text))
    if skeleton(original) != skeleton(prepared):
        raise ValueError("Изменены пробелы, кавычки, дефисы или границы абзацев")
    def gaps(text):
        end = 0
        result = []
        for word in WORD.finditer(text):
            result.append(text[end:word.start()])
            end = word.end()
        return result + [text[end:]]
    for before_gap, after_gap in zip(gaps(original), gaps(prepared)):
        required = [c for c in before_gap if c in ".!?…"]
        available = iter(c for c in after_gap if c in ".!?…")
        if any(not any(c == expected for c in available) for expected in required):
            raise ValueError("Удалена авторская граница предложения")
    if any(c in prepared for c in "<>`") and any(prepared.count(c) != original.count(c) for c in "<>`"):
        raise ValueError("LLM добавила разметку")
    return prepared


def input_identity(manifest):
    return digest([VERSION, ru.VERSION, ru.RULES, manifest.get("book"),
                   manifest.get("paragraph_texts"), manifest.get("voices"),
                   [{k: v for k, v in s.items() if not k.startswith("tts_")} for s in manifest["segments"]]])


def restore_source_words(original, proposed, allowed_numerals):
    """Supertonic-style conservative repair of minor lexical drift.

    Never guess a replacement: put the source word back, without inventing
    its stress. Added/deleted words, numeric changes and large drift stay errors.
    """
    if not isinstance(proposed, str):
        return proposed
    source = list(WORD.finditer(original))
    target = list(WORD.finditer(proposed))
    if len(source) != len(target):
        return proposed
    patches = []
    for index, (a, b) in enumerate(zip(source, target)):
        if bare(a[0]) == bare(b[0]):
            continue
        fa, fb = numeral_family(a[0]), numeral_family(b[0])
        if index in allowed_numerals:
            if fa is None or fa != fb:
                return proposed  # Do not salvage a changed numeric value.
            continue
        if not ru.TOKEN.fullmatch(a[0]) or not ru.TOKEN.fullmatch(b[0]):
            return proposed
        patches.append((b.start(), b.end(), a[0]))
    if len(patches) > max(2, len(source) // 20):
        return proposed
    for start, stop, word in reversed(patches):
        proposed = proposed[:start] + word + proposed[stop:]
    if patches:
        print(f"Сохранён исходный текст: восстановлено слов {len(patches)}; без выдуманных ударений", flush=True)
    return proposed


def prompt_for(manifest, batch, normalized, names, author_texts=None):
    paragraphs = manifest.get("paragraph_texts", {})
    context = []
    for s in batch:
        context.append({"id": s["id"], "speaker": s["speaker"], "paragraph_id": s["paragraph_id"],
                        "paragraph": paragraphs.get(s["paragraph_id"], s["text"])})
    return (ru.RULES["instruction"] +
            "\nВосстанавливай необходимую ё вместо е только по контексту. Существующую ё сохраняй. "
            "Не меняй корректную авторскую пунктуацию. Контекст и роли — только справка, не часть ответа. "
            "Ударения в texts предложены словарями: проверь их по контексту и исправляй неверные. "
            "Только ударения из author_texts являются авторскими и должны быть сохранены точно. "
            "Каждый элемент texts соответствует ровно одной неизменяемой роли; ответ только {\"texts\":[...]}.\n" +
            json.dumps({"texts": normalized, "author_texts": author_texts or normalized, "names": names, "context": context,
                        "book": manifest["book"].get("title"),
                        "cast": [{"id": v["id"], "name": v.get("name"), "gender": v.get("voice_gender")} for v in manifest["voices"]]}, ensure_ascii=False))


def learn_names(texts, names):
    for text in texts:
        for m in ru.TOKEN.finditer(text):
            word = m[0]
            # Not sentence-initial capitalization: names inside sentences only.
            preceding = text[:m.start()].rstrip(' «"(—–-')
            if word[0].isupper() and ("\u0301" in word or "ё" in word.lower()) and preceding and preceding[-1] not in ".!?…\n":
                key = bare(word)
                names.setdefault(key, word)
    return names


def check(manifest):
    info = manifest.get("speech_preparation", {})
    if info.get("input_sha256") != input_identity(manifest):
        raise ValueError("Подготовка речи отсутствует/устарела после изменения текста, ролей или нарезки")
    for s in manifest["segments"]:
        normalized = ru.normalize(s["text"])
        if normalized != s.get("tts_normalized"):
            raise ValueError("Нормализация устарела: " + s["id"])
        validate_text(normalized, s.get("tts_text"), stress=False,
                      allowed_numerals=numeral_positions(s["text"], normalized))
        if s.get("tts_sha256") != digest([VERSION, s["text"], s.get("tts_text")]):
            raise ValueError("Нормализованный текст был изменён: " + s["id"])
    return True


def process(manifest, args, out, key=None):
    result = deepcopy(manifest)
    if not result.get("segments"):
        raise ValueError("Пустая книга")
    voice_ids = {v["id"] for v in result["voices"]}
    for s in result["segments"]:
        if s.get("speaker") not in voice_ids:
            raise ValueError("Сначала определите роль: " + s["id"])
    source_hash = input_identity(result)
    dictionary = None
    dictionary_dir = getattr(args, "dictionary_dir", None)
    if dictionary_dir:
        from stress_dictionary import StressDictionary
        dictionary = StressDictionary(dictionary_dir)
    names = {}
    batches, current, size = [], [], 0
    normalized = {}
    dictionary_texts, dictionary_hits = {}, 0
    for s in result["segments"]:
        try:
            target = ru.normalize(s["text"])
        except ValueError as exc:
            raise ValueError(s["id"] + ": " + str(exc)) from None
        if not target:
            raise ValueError("После нормализации фрагмент пуст: " + s["id"])
        normalized[s["id"]] = target
        seeded, hits = dictionary.apply(target) if dictionary else (target, 0)
        validate_text(target, seeded)
        dictionary_texts[s["id"]] = seeded
        dictionary_hits += hits
        if current and size + len(target) > args.batch_chars:
            batches.append(current)
            current, size = [], 0
        current.append(s)
        size += len(target)
    if current:
        batches.append(current)
    if dictionary:
        suffix = "неизвестные и омографы остаются без новых ударений; LLM отключена" if args.command == "offline" else "неизвестные и омографы проверит LLM"
        print(f"Словари Supertonic: {dictionary_hits} слов обработано; {suffix}", flush=True)
    for i, batch in enumerate(batches, 1):
        texts = [normalized[s["id"]] for s in batch]
        if args.command == "offline":
            prepared = [dictionary_texts[s["id"]] for s in batch]
        else:
            def validate(data):
                values = data.get("texts")
                if not isinstance(values, list) or len(values) != len(texts):
                    raise ValueError("Неверное количество элементов texts")
                # A cloud model may restore typography. Canonicalize only dash
                # glyphs again before validation; do not edit words/stresses.
                checked = [validate_text(a, restore_source_words(a, ru.speech_punctuation(b), numeral_positions(s["text"], a)) if isinstance(b, str) else b,
                                         stress=False, allowed_numerals=numeral_positions(s["text"], a))
                           for s, a, b in zip(batch, texts, values)]
                local_names = dict(names)
                for text in checked:
                    for word in ru.TOKEN.findall(text):
                        known = local_names.get(bare(word))
                        if known and word != known:
                            raise ValueError("Изменилось произношение имени: " + bare(word))
                    learn_names([text], local_names)
                return checked
            prepared = obtain(out, f"speech-{i:04d}/{len(batches)}",
                              prompt_for(result, batch, [dictionary_texts[s["id"]] for s in batch],
                                         list(names.values())[-60:], author_texts=texts), validate, args, key)
            if prepared is None:
                return None
            # Missing accents never trigger repair requests or stop synthesis.
            for s, original, text in zip(batch, texts, prepared):
                validate_text(original, text, allowed_numerals=numeral_positions(s["text"], original))
            learn_names(prepared, names)
        for s, text in zip(batch, prepared):
            s["tts_normalized"] = normalized[s["id"]]
            s["tts_dictionary_text"] = dictionary_texts[s["id"]]
            s["tts_text"] = text
            s["tts_sha256"] = digest([VERSION, s["text"], text])
    result["speech_preparation"] = {"version": VERSION, "normalizer": ru.VERSION,
        "mode": "offline" if args.command == "offline" else "llm", "input_sha256": source_hash,
        "model": None if args.command == "offline" else args.model,
        "stress": any("\u0301" in s.get("tts_text", "") for s in result["segments"]), "stress_encoding": "U+0301 after vowel",
        "dictionary_sha256": dictionary.sha256 if dictionary else None,
        "binary_dictionary_entries": dictionary.binary.count if dictionary and dictionary.binary else 0,
        "dictionary_words": dictionary_hits, "missing_stress_policy": "pass unchanged; no retries",
        "names": names, "segments": len(result["segments"])}
    check(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "advance", "offline", "check"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", help="Output manifest (default: update input only after complete validation)")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--endpoint", default="https://ollama.com")
    parser.add_argument("--batch-chars", type=int, default=2400)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--think", dest="no_think", action="store_false", default=True)
    parser.add_argument("--dictionary-dir", default=str(Path(__file__).parent / "data/stress"))
    args = parser.parse_args()
    if args.batch_chars < 100 or args.max_tokens < 1000 or args.timeout <= 0:
        parser.error("Некорректные лимиты")
    if not args.endpoint.startswith("https://"):
        parser.error("Для передачи ключа требуется HTTPS")
    try:
        manifest = load(args.manifest)
        if args.command == "check":
            check(manifest)
            print("OK: подготовка речи соответствует исходному тексту и ролям")
            return
        destination = Path(args.out or args.manifest)
        cache = destination.parent / "speech-cache"
        existing = manifest.get("speech_preparation", {})
        from stress_dictionary import StressDictionary
        dictionary_hash = StressDictionary(args.dictionary_dir).sha256
        wanted = "offline" if args.command == "offline" else "llm"
        if existing.get("mode") == wanted and existing.get("dictionary_sha256") == dictionary_hash and (wanted == "offline" or existing.get("model") == args.model):
            try:
                check(manifest)
            except ValueError:
                pass
            else:
                if destination.resolve() != Path(args.manifest).resolve():
                    write(destination, manifest)
                print("OK: уже подготовленный текст проверен; повторные запросы не нужны")
                return
        key = (os.environ.get("OLLAMA_API_KEY") or getpass.getpass("Ollama API key (не сохраняется): ")) if args.command == "run" else None
        result = process(manifest, args, cache, key)
        if result is None:
            return
        write(destination, result)
        pending = cache / "pending.json"
        if pending.exists():
            pending.unlink()
        print(json.dumps({k: v for k, v in result["speech_preparation"].items() if k != "names"}, ensure_ascii=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Ошибка подготовки речи: {exc}\n")


if __name__ == "__main__":
    main()
