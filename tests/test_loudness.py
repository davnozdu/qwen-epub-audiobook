import unittest
from normalize_book_audio import gains_for


class LoudnessTests(unittest.TestCase):
    def test_target_is_common_and_preserves_peak_headroom(self):
        measurements = {"a": {"lufs": -28, "true_peak_db": -8}, "b": {"lufs": -18, "true_peak_db": -4}}
        target, gains = gains_for(measurements)
        for key, value in measurements.items():
            self.assertAlmostEqual(value["lufs"] + gains[key], target)
            self.assertLessEqual(value["true_peak_db"] + gains[key], -1.5)

    def test_silent_role_is_not_amplified(self):
        _, gains = gains_for({"speech": {"lufs": -20, "true_peak_db": -4},
                              "silence": {"lufs": float('-inf'), "true_peak_db": float('-inf')}})
        self.assertEqual(gains['silence'], 0)
        with self.assertRaises(ValueError):
            gains_for({"silent": {"lufs": float('-inf'), "true_peak_db": float('-inf')}})
