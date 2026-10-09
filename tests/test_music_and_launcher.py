import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_audiobook import input_path, prepare_roles
from setup_audiobook import valid_model, ensure_model
from mix_book_music import mix


class KitTests(unittest.TestCase):
    def test_dragged_path_and_book_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'books').mkdir()
            file=root/'books/Книга с пробелом.epub';file.touch()
            self.assertEqual(input_path('Книга с пробелом.epub',root,True),file.resolve())
            self.assertEqual(input_path('"'+str(file)+'"',root,True),file.resolve())

    def test_missing_and_corrupt_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            spec={'files':[{'path':'model.bin','size':3,'sha256':hashlib.sha256(b'yes').hexdigest()}]}
            self.assertFalse(valid_model(root,spec))
            (root/'model.bin').write_bytes(b'yes');self.assertTrue(valid_model(root,spec))
            (root/'model.bin').write_bytes(b'bad');self.assertFalse(valid_model(root,spec))

    def test_no_silent_single_voice_fallback(self):
        with self.assertRaisesRegex(ValueError,'рассказчика запрещено'):
            prepare_roles(Path('missing.epub'),Path('.'),False,None)

    def test_download_is_saved_inside_kit_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'kit'
            spec={'repo':'example/model','revision':'fixed','files':[{'path':'config.json','size':3,
                  'sha256':hashlib.sha256(b'yes').hexdigest()}]}
            def download(command,check):
                self.assertEqual(command[0],'curl')
                destination=Path(command[-1]);destination.write_bytes(b'yes')
            with patch('setup_audiobook.subprocess.run',side_effect=download):
                target=ensure_model(root,'TestModel',spec)
            self.assertEqual(target,root/'models/TestModel')
            self.assertEqual((target/'config.json').read_bytes(),b'yes')

    def test_corrupt_file_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'models/TestModel';target.mkdir(parents=True)
            (target/'config.json').write_bytes(b'bad')
            spec={'repo':'example/model','revision':'fixed','files':[{'path':'config.json','size':3,
                  'sha256':hashlib.sha256(b'yes').hexdigest()}]}
            with patch('setup_audiobook.subprocess.run') as download:
                with self.assertRaisesRegex(ValueError,'не перезаписываю'): ensure_model(root,'TestModel',spec)
                download.assert_not_called()
            self.assertEqual((target/'config.json').read_bytes(),b'bad')

    def test_music_loops_and_voice_gain_is_one(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'speech';source.mkdir()
            sf.write(source/'sample.wav',np.full(24000,.1,dtype=np.float32),24000,subtype='FLOAT')
            sf.write(root/'music.wav',np.full(2400,.2,dtype=np.float32),24000,subtype='FLOAT')
            (source/'benchmark.json').write_text(json.dumps({'state':'complete','final_audio_seconds':1}))
            (source/'timeline.json').write_text('{"segments":[]}')
            report=mix(source,root/'mixed',root/'music.wav',5)
            samples,sr=sf.read(root/'mixed/sample.wav')
            self.assertEqual(len(samples),24000)
            np.testing.assert_allclose(samples,.11,atol=.00004)
            self.assertEqual(report['speech_gain'],1)
            self.assertTrue(report['loop'])
            self.assertEqual((source/'timeline.json').read_bytes(),(root/'mixed/timeline.json').read_bytes())
