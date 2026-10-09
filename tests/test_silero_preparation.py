import unittest
from unittest.mock import patch
from prepare_silero import acute, project, hybrid_text, choose_cloud


class SileroPreparationTest(unittest.TestCase):
    def test_cloud_no_does_not_read_key(self):
        for answer in ['n', 'N', 'нет']:
            with patch('builtins.input', return_value=answer), patch('prepare_silero.getpass.getpass') as key:
                self.assertEqual(choose_cloud(), (False, None))
                key.assert_not_called()

    def test_cloud_enter_or_yes_reads_hidden_key(self):
        for answer in ['', 'Y', 'да']:
            with patch('builtins.input', return_value=answer), patch('prepare_silero.getpass.getpass', return_value=' test-credential ') as key:
                self.assertEqual(choose_cloud(), (True, 'test-credential'))
                key.assert_called_once()

    def test_cloud_empty_key_skips_llm(self):
        with patch('builtins.input', return_value=''), patch('prepare_silero.getpass.getpass', return_value=' '):
            self.assertEqual(choose_cloud(), (False, None))

    def test_cloud_invalid_answer_reasks(self):
        with patch('builtins.input', side_effect=['wrong', 'нет']), patch('prepare_silero.getpass.getpass') as key:
            self.assertEqual(choose_cloud(), (False, None))
            key.assert_not_called()

    def test_cloud_eof_skips_llm(self):
        with patch('builtins.input', side_effect=EOFError), patch('prepare_silero.getpass.getpass') as key:
            self.assertEqual(choose_cloud(), (False, None))
            key.assert_not_called()

    def test_encoding(self):
        self.assertEqual(acute('был+а'), 'была́')
        self.assertEqual(acute('+ёлка'), 'ёлка')

    def test_preserve_text(self):
        self.assertEqual(project('Она была. — Всё!', 'он+а был+а. всё!'), 'Она́ была́. — Всё!')

    def test_author_accent(self):
        self.assertEqual(project('зво́нит', 'звон+ит'), 'зво́нит')

    def test_no_one_vowel_marks(self):
        self.assertEqual(project('На полке.', 'н+а п+олке.'), 'На по́лке.')

    def test_word_changes_rejected(self):
        with self.assertRaises(ValueError):
            project('Она была.', 'Он б+ыл.')

    def test_bad_marks(self):
        with self.assertRaises(ValueError):
            acute('бы+ла')

    def test_hybrid_priority(self):
        text, sources = hybrid_text('Она села у замка.', 'Она́ се́ла у замка.',
                                    'Она́ села́ у за́мка.', {'замка'})
        self.assertEqual(text, 'Она́ се́ла у за́мка.')
        self.assertEqual(sources, ['dictionary', 'dictionary', 'unchanged', 'silero_homograph'])

    def test_hybrid_unknown(self):
        text, sources = hybrid_text('Неологизм.', 'Неологизм.', 'Неологи́зм.', set())
        self.assertEqual(text, 'Неологи́зм.')
        self.assertEqual(sources, ['silero_unknown'])

    def test_hybrid_preserves_author(self):
        text, _ = hybrid_text('зво́нит ёлка', 'зво́нит ёлка', 'звони́т ёлка', {'звонит'})
        self.assertEqual(text, 'зво́нит ёлка')
