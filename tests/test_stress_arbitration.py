from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import arbitrate_stress as a
import prepare_speech as p
from prepare_qwen import digest, request_spec


def fixture():
    text = 'Она была. Потом зво́нит.'
    s = {'id': 's1', 'paragraph_id': 'p1', 'text': text, 'speaker': 'narrator', 'pause_after_ms': 800,
         'tts_normalized': text, 'tts_dictionary_text': 'Она́ бы́ла. По́том зво́нит.',
         'tts_silero_text': 'Она́ была́. По́том звони́т.', 'tts_text': 'Она́ бы́ла. По́том зво́нит.'}
    m = {'book': {}, 'voices': [{'id': 'narrator'}], 'segments': [s], 'paragraph_texts': {'p1': text}}
    s['tts_sha256'] = digest([p.VERSION, text, s['tts_text']])
    m['speech_preparation'] = {'version': p.VERSION, 'input_sha256': p.input_identity(m)}
    p.check(m)
    return m


class ArbitrationTest(unittest.TestCase):
    def test_only_disagreement_not_authored(self):
        rows = a.disputes(fixture())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['word_index'], 1)
        self.assertEqual(set(rows[0]['candidates'].values()), {'бы́ла', 'была́'})
        self.assertIn('Она была.', rows[0]['context'])

    def test_selection_changes_only_accent(self):
        m = fixture(); rows = a.disputes(m)
        result, audit = a.apply(m, rows, {'decisions': [{'id': rows[0]['id'], 'choice': 'B', 'reason': 'Норма'}]})
        self.assertEqual(result['segments'][0]['tts_text'], 'Она́ была́. По́том зво́нит.')
        self.assertEqual(result['segments'][0]['pause_after_ms'], 800)
        self.assertEqual(result['segments'][0]['speaker'], 'narrator')
        self.assertEqual(m['segments'][0]['tts_text'], 'Она́ бы́ла. По́том зво́нит.')

    def test_unknown_keeps_hybrid(self):
        m = fixture(); rows = a.disputes(m)
        result, _ = a.apply(m, rows, {'decisions': [{'id': rows[0]['id'], 'choice': 'unknown', 'reason': 'Не уверен'}]})
        self.assertEqual(result['segments'], m['segments'])

    def test_rejects_rewriting_and_missing_ids(self):
        rows = a.disputes(fixture())
        for data in [{'texts': ['Новый текст']}, {'decisions': []},
                     {'decisions': [{'id': rows[0]['id'], 'choice': 'C', 'reason': 'Третий вариант'}]}]:
            with self.assertRaises(ValueError): a.validate(rows, data)

    def test_no_disputes_no_request(self):
        m = fixture(); m['segments'][0]['tts_silero_text'] = m['segments'][0]['tts_dictionary_text']
        args = SimpleNamespace(model='fast')
        with patch.object(a, 'obtain') as mocked:
            result, report = a.arbitrate(m, args, Path('.'))
        mocked.assert_not_called()
        self.assertEqual(report['disputes'], 0)

    def test_bad_service_keeps_hybrid(self):
        m = fixture()
        with patch.object(a, 'obtain', side_effect=ValueError('offline')):
            result, report = a.arbitrate(m, SimpleNamespace(model='fast'), Path('.'))
        self.assertEqual(result['segments'], m['segments'])
        self.assertTrue(report['errors'])

    def test_no_thinking_request(self):
        spec = request_spec(a.prompt_for(a.disputes(fixture())), SimpleNamespace(model='fast', no_think=True, max_tokens=4096))
        self.assertFalse(spec['think'])
        self.assertEqual(spec['options']['temperature'], 0)
