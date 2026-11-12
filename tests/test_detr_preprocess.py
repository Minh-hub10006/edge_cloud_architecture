"""cv2 DETR preprocess matches the PIL baseline closely and keeps the tensor contract."""
import unittest

import numpy as np

from inference.cloud_detr import CloudDetrAdapter
from scripts.ab_detr_preprocess import match


def frame():
    rng = np.random.default_rng(0)
    image = np.dstack([np.tile(np.linspace(40, 200, 1920, dtype=np.float32), (1080, 1))] * 3)
    for _ in range(60):
        x, y = rng.integers(0, 1800), rng.integers(0, 1000)
        w, h = rng.integers(30, 200, 2)
        image[y:y + h, x:x + w] = rng.integers(0, 255, 3)
    return np.clip(image + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)


def prepare(kind, mode, image):
    adapter = object.__new__(CloudDetrAdapter)
    adapter.kind, adapter.preprocess = kind, mode
    return adapter.prepare(image)


class DetrPreprocessTests(unittest.TestCase):
    def test_cv2_matches_pil_contract_and_values(self):
        image = frame()
        for kind, name, scale in (("rtdetrv2_small", "images", 1.0), ("lwdetr_large", "input", .226)):
            old, new = prepare(kind, "pil", image), prepare(kind, "cv2", image)
            self.assertEqual(set(old), set(new))
            a, b = old[name], new[name]
            self.assertEqual((b.shape, b.dtype), ((1, 3, 640, 640), np.float32))
            self.assertTrue(b.flags["C_CONTIGUOUS"])
            # Mean difference about 1/255 in [0, 1] pixel units.
            self.assertLess(np.abs(a - b).mean() * scale, 2 / 255)
        sizes = prepare("rtdetrv2_small", "cv2", image)["orig_target_sizes"]
        self.assertEqual(sizes.tolist(), [[1920, 1080]])

    def test_agreement_matcher_is_same_class_greedy(self):
        from contracts import Box
        a = [Box(.1, .1, .3, .3, .9, 2, "car"), Box(.5, .5, .7, .7, .8, 7, "truck")]
        b = [Box(.11, .1, .3, .3, .85, 2, "car"), Box(.5, .5, .7, .7, .8, 5, "bus")]
        pairs = match(a, b)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0][1], .05)


if __name__ == "__main__":
    unittest.main()
