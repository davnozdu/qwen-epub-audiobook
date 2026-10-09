from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import hashlib

try:
    import numpy as np
except ImportError:
    np = None

import synthesize_qwen_mlx as q


class SelectionTests(unittest.TestCase):
    def test_explicit_gender_and_distinct_female_instructions(self):
        def voice(cid, gender, age="adult"):
            return {"id": cid, "voice_gender": gender, "age": age, "voice_description": "Custom timbre"}
        arina = q.reference_instruction(voice("arina", "female"))
        sofya = q.reference_instruction(voice("sofya_ivanovna", "female"))
        self.assertIn("Clearly female feminine voice", arina)
        self.assertNotEqual(arina, sofya)
        self.assertIn("Clearly male masculine voice", q.reference_instruction(voice("gomozov", "male")))
        self.assertIn("Young girl", q.reference_instruction(voice("girl", "female", "child")))
        with self.assertRaises(ValueError):
            q.reference_instruction(voice("unknown", "unknown"))

    def test_temporary_references_deleted_on_success_error_and_interrupt(self):
        for failure in (None, RuntimeError("failed"), KeyboardInterrupt()):
            folders = []
            def fake_render(args, temporary_library):
                folders.append(Path(temporary_library))
                (folders[-1] / "arina.wav").write_bytes(b"test")
                if failure:
                    raise failure
            args = SimpleNamespace(ephemeral_voices=True, voice_library=None,
                                   reference_source=None, import_only=False)
            with patch.object(q, "_render", side_effect=fake_render):
                if failure:
                    with self.assertRaises(type(failure)):
                        q.render(args)
                else:
                    q.render(args)
            self.assertFalse(folders[0].exists())

    def test_permanent_reference_validates_profile_and_audio(self):
        voice = {"id": "narrator", "instruct": "baritone"}
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder)
            self.assertIsNone(q.read_reference(library, voice, "book"))
            wav = library / "narrator.wav"
            wav.write_bytes(b"reference audio")
            q.save(library / "narrator.json", {
                "profile": q.library_profile(voice, "book"), "ref_text": "Reference text",
                "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest()})
            first = q.read_reference(library, voice, "book")
            self.assertEqual(first, q.read_reference(library, voice, "book"))
            with self.assertRaises(ValueError):
                q.read_reference(library, dict(voice, instruct="tenor"), "book")
            wav.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                q.read_reference(library, voice, "book")

    def test_continuous_selection_starts_at_requested_paragraph(self):
        items = [{"paragraph_id": "p1", "id": "s1"}, {"paragraph_id": "p2", "id": "s2"},
                 {"paragraph_id": "p2", "id": "s3"}, {"paragraph_id": "p3", "id": "s4"}]
        self.assertEqual(q.select_segments({"segments": items}, "p2"), items[1:])
        with self.assertRaises(ValueError):
            q.select_segments({"segments": items}, "missing")

    def test_cache_identity_changes_with_voice_or_settings(self):
        original = ["model", "text", "male baritone", 0.7]
        self.assertNotEqual(q.identity(original), q.identity(["model", "text", "female alto", 0.7]))
        self.assertNotEqual(q.identity(original), q.identity(["model", "text", "male baritone", 0.9]))


@unittest.skipIf(np is None, "Requires the separate MLX runtime's numpy")
class AudioTests(unittest.TestCase):
    def test_clone_reuses_reference_instead_of_designing_voice(self):
        calls = []
        result = SimpleNamespace(audio=np.full(2000, 0.1, dtype=np.float32), sample_rate=1000, token_count=20)
        def generate(**kwargs):
            calls.append(kwargs)
            return iter([result])
        model = SimpleNamespace(generate=generate)
        mx = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None), synchronize=lambda: None)
        args = SimpleNamespace(temperature=0.7, max_tokens=1024)
        reference = {"ref_audio": "/saved/narrator.wav", "ref_text": "Текст образца"}
        for text in ("Первая реплика", "Следующая реплика"):
            q.generate_audio(model, mx, text, "Описание", 1, args, reference)
        for call in calls:
            self.assertEqual(call["ref_audio"], reference["ref_audio"])
            self.assertEqual(call["ref_text"], reference["ref_text"])
            self.assertEqual(call["lang_code"], "Russian")
            self.assertNotIn("instruct", call)

    def test_trim_preserves_internal_silence_and_breathing_room(self):
        sr = 1000
        original = np.concatenate([np.zeros(500), np.full(200, 0.1), np.zeros(300), np.full(200, 0.1), np.zeros(500)])
        trimmed = q.trim_silence(original, sr)
        self.assertEqual(len(trimmed), 820)
        self.assertTrue(np.array_equal(trimmed[240:540], np.zeros(300)))
        with self.assertRaises(ValueError):
            q.trim_silence(np.zeros(2000), sr)

    def test_metrics_detect_clipping_and_duration(self):
        data = np.array([0, 1, -1, 0.5], dtype=np.float32)
        metrics = q.audio_metrics(data, 4)
        self.assertEqual(metrics["audio_seconds"], 1)
        self.assertEqual(metrics["clipped_fraction"], 0.5)
        self.assertEqual(metrics["peak"], 1)

    def test_benchmark_uses_wall_time_and_raw_audio_not_montage_pauses(self):
        result = SimpleNamespace(audio=np.full(2000, 0.1, dtype=np.float32), sample_rate=1000, token_count=20)
        model = SimpleNamespace(generate_voice_design=lambda **kwargs: iter([result]))
        mx = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None), synchronize=lambda: None)
        args = SimpleNamespace(temperature=0.7, max_tokens=1024)
        with patch.object(q.time, "perf_counter", side_effect=[10.0, 14.0]):
            _, _, metrics = q.generate_audio(model, mx, "Тест", "Мужской голос", 1, args)
        self.assertEqual(metrics["raw_audio_seconds"], 2)
        self.assertEqual(metrics["synthesis_seconds"], 4)
        self.assertEqual(metrics["rtf"], 2)
        self.assertEqual(metrics["realtime_speed"], 0.5)

    def test_no_audio_or_truncated_audio_is_not_accepted(self):
        mx = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None), synchronize=lambda: None)
        args = SimpleNamespace(temperature=0.7, max_tokens=64)
        for results in ([], [SimpleNamespace(token_count=64)]):
            model = SimpleNamespace(generate_voice_design=lambda **kwargs: iter(results))
            with self.assertRaises(ValueError):
                q.generate_audio(model, mx, "Тест", "Описание", 1, args)


if __name__ == "__main__":
    unittest.main()
