from __future__ import annotations

import unittest

from edge_agent import BlurDifficultyNormalizer


class BlurDifficultyTests(unittest.TestCase):
    def test_blur_variance_maps_to_common_easy_to_hard_scale(self) -> None:
        normalizer = BlurDifficultyNormalizer(blurry_variance=10.0, sharp_variance=110.0)

        self.assertEqual(normalizer.difficulty(10.0), 1.0)
        self.assertAlmostEqual(normalizer.difficulty(60.0), 0.5)
        self.assertEqual(normalizer.difficulty(110.0), 0.0)
        self.assertEqual(normalizer.difficulty(1000.0), 0.0)
        self.assertEqual(normalizer.difficulty(0.0), 1.0)

    def test_invalid_blur_calibration_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BlurDifficultyNormalizer(blurry_variance=100.0, sharp_variance=50.0)


if __name__ == "__main__":
    unittest.main()
