"""Иностранные вставки латиницей → русское прочтение для модели (без сети: LLM подменена)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import foreign_reading as f  # noqa: E402


class FakeLLM:
    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def call(self, prompt):
        self.calls += 1
        items = json.loads(prompt[prompt.index('{"items"'):])["items"]
        return {"readings": [{"id": item["id"], "reading": self.answers.get(item["text"], "")} for item in items]}


class ForeignReadingTest(unittest.TestCase):
    def test_phrases_are_found_whole(self):
        self.assertEqual(f.fragments("Он сказал: ex post facto, а потом — Übermensch и д’Артаньян."),
                         ["ex post facto", "Übermensch"])

    def test_reading_rules(self):
        self.assertTrue(f.valid("Ги́ллиан Флинн"))
        self.assertTrue(f.valid("экс пост фа́кто"))
        self.assertFalse(f.valid("Гиллиан"))                  # два слога без ударения
        self.assertFalse(f.valid("фла́йн"))                   # односложное с ударением
        self.assertFalse(f.valid("Gillian"))                  # не кириллица
        self.assertFalse(f.valid("ги́лли́ан"))                 # два ударения

    def test_manifest_gets_readings_only_in_text_for_the_model(self):
        work = Path(tempfile.mkdtemp())
        manifest = {"segments": [
            {"id": "s1", "text": "Gillian Flynn. THE GROWNUP.", "tts_text": "Gillian Flynn. THE GROWNUP."},
            {"id": "s2", "text": "Он сказал: ex post facto.", "tts_text": "Он сказа́л: ex post facto."}]}
        path = work / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        llm = FakeLLM({"Gillian Flynn": "Ги́ллиан Флинн", "THE GROWNUP": "зе гро́унап",
                       "ex post facto": "экс пост facto"})              # последнее — неверный ответ
        f.prepare(path, llm, log=lambda message: None)
        result = json.loads(path.read_text(encoding="utf-8"))["segments"]
        self.assertEqual(result[0]["tts_text"], "Ги́ллиан Флинн. зе гро́унап.")
        self.assertEqual(result[0]["text"], "Gillian Flynn. THE GROWNUP.")       # текст книги не тронут
        self.assertEqual(result[1]["tts_text"], "Он сказа́л: ex post facto.")    # неверное прочтение не взято
        f.prepare(path, llm, log=lambda message: None)
        self.assertEqual(llm.calls, 2)        # повтор спросил только то, что не получилось в первый раз

    def test_without_llm_text_stays_as_written(self):
        work = Path(tempfile.mkdtemp())
        path = work / "manifest.json"
        path.write_text(json.dumps({"segments": [{"id": "s1", "text": "Copyright", "tts_text": "Copyright"}]}),
                        encoding="utf-8")
        f.prepare(path, None, log=lambda message: None)
        self.assertEqual(json.loads(path.read_text())["segments"][0]["tts_text"], "Copyright")


if __name__ == "__main__":
    unittest.main()
