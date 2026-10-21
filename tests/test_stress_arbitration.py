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


def yo_fixture():
    """Словарь и Silero спорят о ё: «все» или «всё» — решает смысл предложения."""
    text = 'Он узнал все. Потом ушел.'
    s = {'id': 's1', 'paragraph_id': 'p1', 'text': text, 'speaker': 'narrator', 'pause_after_ms': 800,
         'tts_normalized': text, 'tts_dictionary_text': 'Он узна́л все. По́том ушёл.',
         'tts_silero_text': 'Он узна́л всё. По́том ушёл.', 'tts_text': 'Он узна́л все. По́том ушёл.'}
    m = {'book': {}, 'voices': [{'id': 'narrator'}], 'segments': [s], 'paragraph_texts': {'p1': text}}
    s['tts_sha256'] = digest([p.VERSION, text, s['tts_text']])
    m['speech_preparation'] = {'version': p.VERSION, 'input_sha256': p.input_identity(m)}
    p.check(m)
    return m


class YoArbitrationTest(unittest.TestCase):
    def test_yo_disagreement_is_a_dispute(self):
        rows = a.disputes(yo_fixture())
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]['candidates'].values()), {'все', 'всё'})

    def test_yo_option_is_shown_stressed(self):
        prompt = a.prompt_for(a.disputes(yo_fixture()))
        self.assertIn('"display": "всЁ"', prompt)
        self.assertIn('"current_stress": 1', prompt)
        self.assertIn('"yo": true', prompt)

    def test_choosing_yo_changes_only_that_letter(self):
        m = yo_fixture(); rows = a.disputes(m)
        label = next(k for k, v in rows[0]['candidates'].items() if v == 'всё')
        result, _ = a.apply(m, rows, {'decisions': [{'id': rows[0]['id'], 'choice': label, 'reason': 'Смысл: всё'}]})
        self.assertEqual(result['segments'][0]['tts_text'], 'Он узна́л всё. По́том ушёл.')

    def test_different_words_are_not_a_dispute(self):
        m = yo_fixture(); m['segments'][0]['tts_silero_text'] = 'Он узна́л вся. По́том ушёл.'
        self.assertEqual(a.disputes(m), [])


class ProviderClientTest(unittest.TestCase):
    """Арбитр через тот же клиент, что разметка ролей: Ollama Cloud или DeepSeek API."""

    def answer(self, body, choice):
        import json, re
        ids = re.findall(r'"id": "(d\d+)"', body['messages'][-1]['content'])
        content = json.dumps({'decisions': [{'id': i, 'choice': choice, 'reason': 'норма'} for i in ids]})
        if 'response_format' in body:
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
        return {'done_reason': 'stop', 'message': {'content': content}}

    def run_with(self, provider, choice='B'):
        import tempfile
        from cast_book import LLM
        sent = []
        def transport(body):
            sent.append(body)
            return self.answer(body, choice)
        with tempfile.TemporaryDirectory() as cache:
            llm = LLM('key', 'deepseek-v4.1-flash', cache, provider=provider, transport=transport)
            result, report = a.arbitrate(fixture(), SimpleNamespace(model='deepseek-v4.1-flash'), Path(cache), 'key', llm)
        return result, report, sent

    def test_ollama_client(self):
        result, report, sent = self.run_with('ollama')
        self.assertEqual(len(sent), 1)
        self.assertFalse(sent[0]['think'])
        self.assertEqual(report['disputes'], 1)
        self.assertEqual(report['errors'], [])

    def test_deepseek_client(self):
        result, report, sent = self.run_with('deepseek')
        self.assertEqual(sent[0]['thinking'], {'type': 'disabled'})
        self.assertEqual(report['errors'], [])

    def test_rejected_key_keeps_hybrid_and_stops(self):
        import tempfile, urllib.error
        from cast_book import LLM
        def transport(body):
            raise urllib.error.HTTPError('https://ollama.com', 401, 'unauthorized', {}, None)
        m = fixture()
        with tempfile.TemporaryDirectory() as cache:
            llm = LLM('bad', 'm', cache, transport=transport)
            result, report = a.arbitrate(m, SimpleNamespace(model='m'), Path(cache), 'bad', llm)
        self.assertEqual([s['tts_text'] for s in result['segments']], [s['tts_text'] for s in m['segments']])
        self.assertIn('отклонил ключ', report['errors'][0])
