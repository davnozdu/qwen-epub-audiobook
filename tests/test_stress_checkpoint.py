import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_stress_qwen as setup


class StressCheckpointTests(unittest.TestCase):
    def test_pinned_source_is_merged_checkpoint(self):
        self.assertEqual(setup.SPEC["repo"], "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf")
        main = next(f for f in setup.SPEC["files"] if f["path"] == "model.safetensors")
        self.assertGreater(main["size"], 3_000_000_000)
        self.assertEqual(len(main["sha256"]), 64)

    def test_ready_rejects_wrong_revision(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)
            (target / "stress-source.json").write_text(json.dumps({"repo": setup.SPEC["repo"], "revision": "wrong"}))
            (target / "config.json").write_text(json.dumps({"model_type": "qwen3_tts", "quantization": {"bits": 8}}))
            (target / "model.safetensors").write_bytes(b"x" * 2048)
            (target / "speech_tokenizer").mkdir()
            (target / "speech_tokenizer/model.safetensors").write_bytes(b"test")
            with patch.object(setup, "TARGET", target):
                self.assertFalse(setup.ready())

    def test_missing_conversion_not_ready(self):
        with tempfile.TemporaryDirectory() as d, patch.object(setup, "TARGET", Path(d)):
            self.assertFalse(setup.ready())


if __name__ == "__main__":
    unittest.main()
