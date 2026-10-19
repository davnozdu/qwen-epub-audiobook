from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import russian_text as r
import prepare_speech as p
import audiobook_epub as a
import synthesize_qwen_mlx as q


def accent(text):
    def word(m):
        w = m[0]
        if "ё" in w.lower() or "\u0301" in w or sum(c in r.VOWELS for c in w.lower()) <= 1:
            return w
        i = max(i for i, c in enumerate(w.lower()) if c in r.VOWELS)
        return w[:i+1] + "\u0301" + w[i+1:]
    return r.TOKEN.sub(word, text)


def manifest():
    text = "Арина ждала 5 минут. Гомозов пришёл в 02:01."
    segments = []
    cut = text.index(". ") + 1
    for i, (start, stop, speaker) in enumerate([(0, cut, "arina"), (cut + 1, len(text), "gomozov")]):
        segments.append({"id": "s" + str(i), "order": i, "speaker": speaker, "voice_id": speaker,
            "text": text[start:stop], "paragraph_id": "p1", "chapter_id": "c1", "pause_after_ms": 150,
            "source": {"char_start": start, "char_end": stop, "href": "chapter.xhtml", "block_index": 0}})
    return {"book": {"title": "Тест", "input_sha256": "test", "author": "Автор"},
        "voices": [{"id": "arina", "name": "Арина", "voice_gender": "female"},
                   {"id": "gomozov", "name": "Гомозов", "voice_gender": "male"}],
        "paragraph_texts": {"p1": text}, "chapters": [{"id": "c1", "title": "Глава 1"}], "segments": segments}


def args(mode="offline", size=2400):
    return SimpleNamespace(command=mode, batch_chars=size, model="deepseek-v4.1-flash",
        no_think=True, endpoint="https://ollama.com", max_tokens=16000, timeout=30)


class NormalizerTests(unittest.TestCase):

    def test_dashes_never_reach_the_model(self):
        """Тире из текста для модели убираются, а текст книги остаётся как написан.

        Модель спотыкается на тире: «Смерть – высокая особа». В русской речи тире не
        произносится, это знак паузы, и паузу дают запятая и точка. Поэтому в копии для
        синтеза тире нет, а отображаемый текст книги не меняется никогда.
        """
        self.assertEqual(r.normalize("Смерть – высокая особа.", strict=False), "Смерть высокая особа.")
        self.assertEqual(r.normalize("руке, – поза напряженная.", strict=False), "руке, поза напряженная.")
        self.assertEqual(r.normalize("– Да, – сказал он. — Нет.", strict=False), "Да, сказал он. Нет.")
        # Дефис внутри слова — часть слова, его не трогаем; тире внутри слова становится дефисом.
        # Дефисные слова склеиваются в одно слово: модель читает их без запинки посреди слова.
        self.assertEqual(r.normalize("Солдат-пехотинец по–моему.", strict=False), "Солдатпехотинец помоему.")
        self.assertEqual(r.normalize("как-будто как-нибудь", strict=False), "какбудто какнибудь")
        # А даты и диапазоны — это числа, их дефис остаётся.
        self.assertEqual(r.speech_punctuation("дата 2024-05-03 в тексте"), "дата 2024-05-03 в тексте")
        # Диапазон чисел — это не пауза: дефис остаётся.
        self.assertEqual(r.speech_punctuation("страницы 5–10"), "страницы 5-10")
        # Проверка подготовки речи на таком тексте проходит: слова и границы предложений целы.
        original = r.normalize("Смерть – высокая особа, и это заметно.", strict=False)
        p.validate_text(original, original)

    def test_displayed_book_text_keeps_its_dashes(self):
        """Правка касается только копии для модели: в книге и в разметке тире остаются."""
        text = "Смерть – высокая особа."
        source = {"title": "Тест", "author": "", "input_sha256": "hash", "normalization": "test",
                  "paragraphs": [{"id": "p1", "chapter_id": "ch001", "text": text,
                                  "source": {"href": "a.xhtml", "block_index": 0, "element_id": None, "tag": "p"}}],
                  "chapters": [{"id": "ch001", "href": "a.xhtml", "title": "Глава"}]}
        from types import SimpleNamespace
        import prepare_qwen as pq
        labels = [{"id": "p1", "spans": [{"start": 0, "end": len(text), "speaker": "narrator",
                                         "confidence": "high", "reason": ""}]}]
        args = SimpleNamespace(target_chars=220, max_chars=260, min_chars=100, model="test", no_think=False)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            pq.export_project(out, source, {"characters": [
                {"id": "narrator", "name": "Автор", "gender": "male", "age": "adult", "aliases": [],
                 "evidence": "Повествование", "voice_description": "Calm adult male baritone."}]},
                labels, args)
            manifest = pq.load(out / "manifest.json")
        self.assertEqual(manifest["segments"][0]["text"], text)          # в разметке тире на месте
        self.assertNotIn("–", r.normalize(manifest["segments"][0]["text"], strict=False))

    def test_binary_dictionary_utf8_lookup_and_close(self):
        import struct
        from stress_dictionary import BinaryAccentDictionary
        pairs = sorted({"ветер": "в+етер", "молоденький": "мол+оденький", "ёж": "+ёж"}.items(), key=lambda x: x[0].encode())
        offsets, data = [], bytearray()
        for key, value in pairs:
            k, v = key.encode(), value.encode()
            offsets.append(len(data))
            data.extend(struct.pack("<HH", len(k), len(v)) + k + v)
        encoded = struct.pack("<4sIIQQ", b"SACC", 1, len(pairs), 28, 28+4*len(pairs)) + struct.pack("<3I", *offsets) + data
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.sacc"
            path.write_bytes(encoded)
            dictionary = BinaryAccentDictionary(path)
            try:
                self.assertEqual(dictionary.lookup("Молоденький"), "мол+оденький")
                self.assertEqual(dictionary.lookup("ёж"), "+ёж")
                self.assertIsNone(dictionary.lookup("неизвестно"))
            finally:
                dictionary.close()
            self.assertIsNone(dictionary.buffer)
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                BinaryAccentDictionary(path)

    def test_supertonic_reference_cases(self):
        cases = {
            "В списке 1001 имя и 1101 запись.": "В списке одна тысяча одно имя и одна тысяча сто одна запись.",
            "1 101": "одна тысяча сто один", "без 25 записей": "без двадцати пяти записей",
            "к 234": "к двумстам тридцати четырём", "2 000 кг.": "две тысячи килограммов.",
            "1 500-й раз": "тысяча пятисотый раз", "05.05.2024": "Пятого мая две тысячи двадцать четвёртого года",
            "02:01": "два часа одна минута", "в 2024 году": "в две тысячи двадцать четвёртом году",
            "21-я глава": "двадцать первая глава", "IV век": "четвёртый век", "до 1 км": "до одного километра",
            "без 2 ₽": "без двух рублей", "с 5 кг": "с пятью килограммами", "к 20 м": "к двадцати метрам",
            "$100": "сто долларов", "3 мая": "Третьего мая", "1/3": "одна третья", "3,14": "три целых четырнадцать сотых",
            "2 км 5 кг 1%": "два километра пять килограммов один процент",
            "+7 (912) 345-67-89": "плюс семь девятьсот двенадцать триста сорок пять шестьдесят семь восемьдесят девять",
            "т.е. ФСБ": "то есть эф эс бэ", "зелё-\nный лес [1]": "зелёный лес",
            "20 августа 1991 года": "Двадцатого августа тысяча девятьсот девяносто первого года",
            "к 1 сентября": "к первому сентября", "на 20 августа": "на двадцатое августа",
            "1941–1945 гг.": "тысяча девятьсот сорок первого тысяча девятьсот сорок пятого годов.",
            "1813-14 годов": "тысяча восемьсот тринадцатого тысяча восемьсот четырнадцатого годов",
            "война 1941–45 гг.": "война тысяча девятьсот сорок первого тысяча девятьсот сорок пятого годов.",
            "Глава 3": "Глава третья", "Том II": "Том второй", "IV.": "Глава четвёртая.",
            "Пётр I": "Пётр Первый", "при Петре I": "при Петре Первом", "главы с III по V": "главы с третьей по пятую",
            "Тома I и II": "Тома первый и второй", "Читай главы III и IV": "Читай главы третью и четвёртую",
            "-5°C": "минус пять градусов Цельсия", "2,01": "две целых одна сотая",
            "светло́ и берёза": "светло́ и берёза", "Письмо: test@example.com": "Письмо: test@example.com",
            "https://example.com/a/2024": "https://example.com/a/2024",
            "22 кровати": "двадцать две кровати", "21 книгу": "двадцать одну книгу",
            "к 1 книге": "к одной книге", "2 мужчины": "два мужчины",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(r.normalize(original), expected)
                self.assertEqual(r.normalize(expected), expected)

    def test_invalid_and_unsupported_formats_fail_closed(self):
        for text in ["31.02.2024", "25:10", "Цена 1234567890123 руб.", "2024-05-03", "1/0", "2024-05"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                r.normalize(text)

    def test_unsupported_number_does_not_kill_the_whole_book(self):
        """Одно странное число не должно ронять двухчасовую книгу: в мягком режиме его показывают."""
        pending = []
        self.assertEqual(r.normalize("дата 2024-05-03 в тексте", strict=False, unresolved=pending),
                         "дата 2024-05-03 в тексте")
        self.assertEqual(pending, ["2024-05-03"])
        with self.assertRaises(ValueError) as failure:      # строгий режим остаётся строгим
            r.normalize("дата 2024-05-03 в тексте")
        self.assertIn("2024-05-03", str(failure.exception))  # и называет виновника, а не молчит

    def test_numeric_value_and_range(self):
        self.assertEqual(r.normalize("10–15 км"), "от десяти до пятнадцати километров")
        self.assertEqual(r.normalize("в 1500 метрах"), "в одной тысяче пятистах метрах")
        self.assertEqual(r.integer(999999999999).split()[0], "девятьсот")


class StressTests(unittest.TestCase):
    def test_minor_lexical_repair_restores_source_without_guessing_accents(self):
        self.assertEqual(p.restore_source_words("Она оборванная.", "Она́ оборо́ванная.", set()), "Она́ оборванная.")
        self.assertEqual(p.restore_source_words("Пять рублей.", "Шесть рубле́й.", {0}), "Шесть рубле́й.")
        self.assertEqual(p.restore_source_words("Она здесь.", "Она́ ти́хо здесь.", set()), "Она́ ти́хо здесь.")

    def test_missing_stress_does_not_stop_book_or_cause_repair_requests(self):
        calls = []
        def obtain(out, name, prompt, validator, settings, key):
            calls.append(name)
            payload = __import__("json").loads(prompt.split("\n")[-1])
            return validator({"issues": []})
        with tempfile.TemporaryDirectory() as temp, patch.object(p, "obtain", side_effect=obtain):
            result = p.process(manifest(), args("run"), Path(temp), "test")
        self.assertEqual(len(calls), 1)
        self.assertTrue(p.check(result))
        self.assertEqual(p.validate_text("молоденький", "молоденький"), "молоденький")
        self.assertEqual(p.validate_text("кто-нибудь", "кто-нибудь"), "кто-нибудь")

    def test_supertonic_dictionary_precedes_llm_and_author_stress_is_protected(self):
        from stress_dictionary import StressDictionary
        with tempfile.TemporaryDirectory() as folder:
            p.write(Path(folder) / "dictionary.json", {"exceptions": {"его": [2, -1]},
                "corrections": {"ветер": "в+етер"}, "homographs": {"замок": ["з+амок", "зам+ок"]},
                "prefixes": {"темно": "тёмно"}, "phrases": {"замок": [["дверной замок", "зам+ок"]]}})
            dictionary = StressDictionary(folder)
            self.assertEqual(dictionary.apply("Ветер его, замок, светло́, тёмно.")[0], "Ве́тер его́, замок, светло́, тёмно.")
            self.assertEqual(dictionary.apply("Дверной замок")[0], "Дверной замо́к")
            self.assertEqual(dictionary.apply("темно-зелёный")[0], "тёмно-зелёный")
            original = manifest()
            original["segments"][0]["text"] = "Ветер его."
            settings = args("run")
            settings.dictionary_dir = folder
            def obtain(out, name, prompt, validator, args, key):
                data = __import__("json").loads(prompt.split("\n")[-1])
                self.assertEqual(data["texts"][0], "Ве́тер его́.")
                self.assertEqual(data["author_texts"][0], "Ветер его.")
                return validator({"issues": []})
            with patch.object(p, "obtain", side_effect=obtain):
                result = p.process(original, settings, Path(folder), "test")
            self.assertTrue(p.check(result))
            self.assertEqual(result["speech_preparation"]["dictionary_words"], 2)

    def test_dashes_are_removed_for_speech_and_epub_text_is_unchanged(self):
        original = manifest()
        original["segments"][0]["text"] = "— Арина – девушка."
        with tempfile.TemporaryDirectory() as temp:
            result = p.process(original, args(), Path(temp))
        self.assertEqual(result["segments"][0]["text"], "— Арина – девушка.")
        self.assertEqual(result["segments"][0]["tts_text"], "Арина девушка.")
        # Универсально: тире для модели не остаётся ни одного, в любом начертании.
        self.assertEqual(r.speech_punctuation("—–‐‑‒−"), "")
        self.assertEqual(r.speech_punctuation("«— О, ещё бы!»"), "«О, ещё бы!»")

    def test_brackets_are_spoken_and_epub_text_is_unchanged(self):
        original = manifest()
        original["segments"][0]["text"] = "Арина (как всегда) пришла."
        with tempfile.TemporaryDirectory() as temp:
            result = p.process(original, args(), Path(temp))
        self.assertEqual(result["segments"][0]["text"], "Арина (как всегда) пришла.")
        self.assertEqual(result["segments"][0]["tts_normalized"],
                         "Арина, скобка открывается, как всегда, скобка закрывается, пришла.")
        self.assertTrue(p.check(result))
        cases = {
            "Он вошёл (тихо).": "Он вошёл, скобка открывается, тихо, скобка закрывается.",
            "(Примечание автора.) Дальше.": "скобка открывается, Примечание автора, скобка закрывается. Дальше.",
            "«(шёпотом) Иди»": "«скобка открывается, шёпотом, скобка закрывается, Иди»",
            "См. [а] и {б}.": "См. скобка открывается, а, скобка закрывается, и, скобка открывается, б, скобка закрывается.",
        }
        for source, spoken in cases.items():
            with self.subTest(source=source):
                self.assertEqual(r.speech_punctuation(source), spoken)

    def test_hyphenated_particles_and_prefixes_have_no_pause(self):
        # Дефис внутри слова модель читает микропаузой: частицы и приставки склеиваются.
        cases = {
            "Кто-то пришёл.": "Ктото пришёл.", "Скажи-ка мне.": "Скажика мне.",
            "Где-нибудь и когда-либо.": "Гденибудь и когдалибо.", "Кое-что случилось.": "Коечто случилось.",
            "По-моему, это по-русски.": "Помоему, это порусски.", "Из-за дождя, из-под стола.": "Изза дождя, изпод стола.",
            "Вот-вот, всё-таки, ну-ка.": "Вотвот, всётаки, нука.",
            "Что\u0301-то упало.": "Что\u0301то упало.",          # авторское ударение перед дефисом
            "Кто\u2011то": "Ктото", "Где\u2013то": "Гдето",        # неразрывный дефис, тире без пробелов
            "ПО-РУССКИ": "порусски", "КТО-ТО ПРИШЁЛ": "ктото пришёл",   # капсом — не аббревиатура
            "ФСБ": "эф эс бэ",                                       # а настоящая аббревиатура — по буквам
            "страницы 5-10": "страницы от пяти до десяти",
        }
        for source, spoken in cases.items():
            with self.subTest(source=source):
                result = r.normalize(source, strict=False)
                self.assertEqual(result, spoken)
                self.assertNotRegex(result, r"[^\W\d_]\u0301?[-\u2010-\u2015]")

    def test_latin_and_signs_are_read_as_russian_words(self):
        cases = {
            "Он сказал: Hello, world!": "Он сказал: хелло, ворлд!",
            "Café Müller": "кафе муллер",
            "Цена 5 % и № 7, § 3.": "Цена пять процентов и номер семь, параграф три.",
            "Том & Джерри": "Том и Джерри",
            "* * *": "",
            "“ок” и „да“": "«ок» и «да»",
            "Пётр I и Людовик XIV": "Пётр Первый и Людовик Четырнадцатый",   # римские — до латиницы
            "Письмо: test@example.com": "Письмо: test@example.com",            # адреса не трогаем
        }
        for source, spoken in cases.items():
            with self.subTest(source=source):
                self.assertEqual(r.normalize(source, strict=False), spoken)

    def test_unstressed_bezo_is_valid_but_real_omissions_are_rejected(self):
        text = "Безо всяких причин."
        target = "Безо вся́ких причи́н."
        self.assertEqual(p.validate_text(text, target), target)
        self.assertEqual(p.validate_text("безо всяких причин", "безо вся́ких причи́н"), "безо вся́ких причи́н")
        for wrong in ["Безо всяких причи́н.", "Безо вся́ких причин.", "Безо вся́ких при́чи́н."]:
            with self.subTest(target=wrong), self.assertRaises(ValueError):
                p.validate_text(text, wrong, stress=True)
        with self.assertRaises(ValueError):
            p.validate_text("оборванная", "оборо́ванная")

    def test_llm_validation(self):
        self.assertEqual(p.validate_text("Она ждала пять минут.", "Она́ ждала́ пять мину́т."), "Она́ ждала́ пять мину́т.")
        self.assertEqual(p.validate_text("владел два домами", "владе́л двумя́ дома́ми"), "владе́л двумя́ дома́ми")
        self.assertEqual(p.validate_text("Все готово.", "Всё гото́во."), "Всё гото́во.")
        for target in ["Она́ ждала́ шесть мину́т.", "Она́ ждала́ пять.", "Она́ тихо́ ждала́ пять мину́т.",
                       "Она ждала пять минут.", "О́на́ ждала́ пять мину́т.", "Он́а ждала́ пять мину́т."]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                p.validate_text("Она ждала пять минут.", target, stress=True)
        with self.assertRaises(ValueError):
            p.validate_text("светло́", "све́тло")
        with self.assertRaises(ValueError):
            p.validate_text("берёза", "бере́за")
        with self.assertRaises(ValueError):
            p.validate_text("Привет, Арина!", "Приве́т, «Ари́на»!")
        with self.assertRaises(ValueError):
            p.validate_text("Да. Нет. Это было не раз.", "Да Нет Э́то бы́ло не раз.")
        with self.assertRaises(ValueError):
            p.validate_text("Автор написал два.", "А́втор написа́л двумя́.",
                            allowed_numerals=p.numeral_positions("Автор написал два.", "Автор написал два."))
        allowed = p.numeral_positions("Владел 2 домами.", "Владел два домами.")
        self.assertEqual(p.validate_text("Владел два домами.", "Владе́л двумя́ дома́ми.",
                         allowed_numerals=allowed), "Владе́л двумя́ дома́ми.")

    def test_processed_text_never_changes_source_roles_or_epub(self):
        original = manifest()
        with tempfile.TemporaryDirectory() as temp:
            result = p.process(original, args(), Path(temp))
            self.assertNotIn("speech_preparation", original)
            self.assertTrue(p.check(result))
            for before, after in zip(original["segments"], result["segments"]):
                self.assertEqual(before, {k: v for k, v in after.items() if not k.startswith("tts_")})
                self.assertEqual(q.speech_text(after), after["tts_text"])
            a.build_text_epub(result, Path(temp) / "test.epub")
            a.validate_epub(Path(temp) / "test.epub", result)
            for field in ["text", "speaker"]:
                edited = deepcopy(result)
                edited["segments"][0][field] += "x"
                with self.assertRaises(ValueError):
                    p.check(edited)
            edited = deepcopy(result)
            edited["segments"][0]["tts_text"] += "x"
            with self.assertRaises(ValueError):
                p.check(edited)

    def test_llm_context_one_request_at_a_time_and_no_partial_publish(self):
        original = manifest()
        calls = []
        def obtain(out, name, prompt, validator, settings, key):
            payload = __import__("json").loads(prompt.split("\n")[-1])
            calls.append(payload)
            return validator({"issues": []})
        with tempfile.TemporaryDirectory() as temp, patch.object(p, "obtain", side_effect=obtain):
            result = p.process(original, args("run", 20), Path(temp), "secret-not-stored")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["context"][0]["paragraph"], original["paragraph_texts"]["p1"])
            self.assertTrue(p.check(result))
            self.assertEqual(result["speech_preparation"]["model"], "deepseek-v4.1-flash")
        with tempfile.TemporaryDirectory() as temp, patch.object(p, "obtain", side_effect=[None]):
            self.assertIsNone(p.process(original, args("advance"), Path(temp)))
            self.assertNotIn("tts_text", original["segments"][0])

    def test_splitting_invalidates_speech_and_timeline_checks_speech(self):
        m = manifest()
        with tempfile.TemporaryDirectory() as temp:
            result = p.process(m, args(), Path(temp))
        split = a.sentence_segments(result)
        self.assertTrue(all("tts_text" not in s for s in split))
        rows = [{"segment_id": s["id"], "text": s["text"], "source": s["source"],
            "speaker": s["speaker"], "tts_text": s["tts_text"], "audio_start": i,
            "audio_end": i + 0.8} for i, s in enumerate(result["segments"])]
        a.verify_timeline(result, rows, 2)
        rows[0]["tts_text"] = "старое аудио"
        with self.assertRaises(ValueError):
            a.verify_timeline(result, rows, 2)
        self.assertEqual(q.speech_text({"text": "original", "tts_text": "светло́"}, "strip"), "светло")


if __name__ == "__main__":
    unittest.main()
