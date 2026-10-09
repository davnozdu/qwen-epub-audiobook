import unittest
from audiobook_epub import boundary_pause


class MontagePauses(unittest.TestCase):
    def test_terminal_punctuation_is_detected_before_quotes_and_dialogue_dashes(self):
        next_row = {"speaker": "narrator", "paragraph_id": "p"}
        for text, ms in [("Текст.", 450), ('«Вопрос?» -', 600), ('«Ответ!» -', 600), ('Мысль… -', 700)]:
            self.assertEqual(boundary_pause({"text": text, "speaker": "narrator", "paragraph_id": "p"}, next_row), ms)

    def test_role_change_inside_sentence_is_not_a_full_sentence_pause(self):
        row = {"text": "сказал автор,", "speaker": "narrator", "paragraph_id": "p"}
        self.assertEqual(boundary_pause(row, {"speaker": "person", "paragraph_id": "p"}), 180)
        self.assertEqual(boundary_pause(row, {"speaker": "person", "paragraph_id": "next"}), 800)
