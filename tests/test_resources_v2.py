"""Edge resource sampler and label-free scene features used by the v2 control state."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from edge_agent.resources_v2 import SMALL_AREA, ResourceSampler, box_features, scene_features


class ResourceTests(unittest.TestCase):
    def test_hottest_readable_zone_and_skips_disabled_sensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (("zone0", "45000"), ("zone1", "71500"), ("zone2", "not-a-number")):
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "temp").write_text(text)
            sampler = ResourceSampler(thermal_glob=str(Path(tmp) / "zone*" / "temp"))
            self.assertEqual(sampler.temperature_c(), 71.5)
            sample = sampler.sample()
        self.assertEqual(set(sample), {"cpu_max_core_pct", "cpu_total_pct", "ram_pct", "temp_c"})
        self.assertGreaterEqual(sample["cpu_max_core_pct"], sample["cpu_total_pct"])
        self.assertEqual(sample["temp_c"], 71.5)

    def test_no_thermal_zone_is_missing_not_zero(self):
        sampler = ResourceSampler(thermal_glob="/definitely/not/here/*")
        self.assertIsNone(sampler.temperature_c())


class SceneTests(unittest.TestCase):
    def test_brightness_contrast_and_motion(self):
        dark = np.zeros((720, 1280, 3), np.uint8)
        bright = np.full((720, 1280, 3), 200, np.uint8)
        first, gray = scene_features(dark)
        self.assertEqual((first["brightness"], first["contrast"]), (0.0, 0.0))
        self.assertNotIn("motion", first)  # no previous sample yet
        second, _ = scene_features(bright, gray)
        self.assertAlmostEqual(second["brightness"], 200, delta=1)
        self.assertAlmostEqual(second["motion"], 200, delta=1)
        self.assertEqual(gray.shape, (90, 160))

    def test_vehicle_count_and_small_ratio(self):
        def box(cls, w, h):
            return SimpleNamespace(class_id=cls, left=.1, top=.1, right=.1 + w, bottom=.1 + h)
        tiny = box(2, .01, .01)          # car smaller than 32x32 px at 1280x720
        large = box(7, .2, .2)           # truck
        person = box(0, .01, .01)        # not a vehicle
        self.assertLess(.01 * .01, SMALL_AREA)
        none = {"conf_mean": None, "uncertain_ratio": None}  # boxes without a confidence (state variant B)
        self.assertEqual(box_features([tiny, large, person]), {"vehicle_count": 2, "small_ratio": .5, **none})
        self.assertEqual(box_features([person]), {"vehicle_count": 0, "small_ratio": None, **none})
        tiny.confidence, large.confidence = .3, .9   # both >= 0.25; one below 0.5
        self.assertEqual(box_features([tiny, large])["uncertain_ratio"], .5)
        self.assertAlmostEqual(box_features([tiny, large])["conf_mean"], .6)


if __name__ == "__main__":
    unittest.main()
