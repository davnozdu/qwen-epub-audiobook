import unittest
from prepare_speech import apply_review, apply_confirmation


class ReviewTests(unittest.TestCase):
    def test_checker_can_only_remove_non_author_stress(self):
        batch = [{"id": "s"}]
        result = apply_review(batch, ["Авторски́й текст был неверен."],
                              ["Авторски́й те́кст был неве́рен."], {"issues": [
                                  {"segment_id": "s", "word_index": 0, "expected_display": None, "reason": "сомнение"},
                                  {"segment_id": "s", "word_index": 1, "expected_display": None, "reason": "сомнение"}]})
        self.assertEqual(result["texts"], ["Авторски́й текст был неве́рен."])
        self.assertEqual(result["review"][0]["action"], "preserved_author")

    def test_same_vowel_cannot_remove_a_correct_mark_even_if_reason_claims_wrong(self):
        result = apply_review([{"id": "s"}], ["Алгоритмы."], ["Алгори́тмы."], {"issues": [
            {"segment_id": "s", "word_index": 0, "expected_display": "алгорИтмы", "reason": "якобы неверно"}]})
        self.assertEqual(result["texts"], ["Алгори́тмы."])
        self.assertEqual(result["review"][0]["action"], "ignored_same_stress")

    def test_contradictory_checks_remove_uncertain_mark_instead_of_forcing_either(self):
        batch = [{"id": "s"}]
        checked = apply_review(batch, ["Слово."], ["Сло́во."], {"issues": [
            {"segment_id": "s", "word_index": 0, "expected_display": "словО", "reason": "ошибка проверяющего"}]})
        result = apply_confirmation(batch, ["Слово."], ["Сло́во."], checked, {"decisions": [
            {"segment_id": "s", "word_index": 0, "choice": "keep"}]})
        self.assertEqual(result["texts"], ["Слово."])
        self.assertEqual(result["review"][0]["action"], "removed_conflicting_reviews")

    def test_rewriting_or_invalid_addresses_are_rejected(self):
        for data in ({"texts": ["Добавленное слово"]}, {"issues": [], "texts": []},
                     {"issues": [{"segment_id": "s", "word_index": True, "reason": "x"}]},
                     {"issues": [{"segment_id": "s", "word_index": 20, "reason": "x"}]},
                     {"issues": [{"segment_id": "s", "word_index": 0, "reason": "x", "word": "подмена"}]}):
            with self.assertRaises(ValueError):
                apply_review([{"id": "s"}], ["Слово."], ["Сло́во."], data)
