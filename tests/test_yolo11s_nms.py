"""Matrix NMS returns exactly the baseline Python greedy NMS selection."""
import unittest

import numpy as np

from yolo11s_contract import nms_matrix, nms_python


def sample(rng):
    rows, scores, labels = [], [], []
    for center, size in zip(rng.uniform(20, 300, (25, 2)), rng.uniform(10, 80, (25, 2))):
        cls = int(rng.integers(0, 3))
        for _ in range(int(rng.integers(5, 15))):
            c = center + rng.normal(0, size * .08, 2)
            s = size * rng.uniform(.85, 1.15, 2)
            rows.append(np.r_[c - s / 2, c + s / 2])
            scores.append(rng.uniform(.05, .95))
            labels.append(cls if rng.random() < .8 else int(rng.integers(0, 3)))
    n = 2100 - len(rows)
    xy, wh = rng.uniform(0, 300, (n, 2)), rng.uniform(2, 60, (n, 2))
    boxes = np.vstack([np.array(rows), np.c_[xy, xy + wh]]).astype(np.float32)
    values = np.r_[scores, rng.uniform(0, .105, n)].astype(np.float32)
    values[rng.integers(0, len(rows), 10)] = np.float32(.5)  # score ties
    return boxes, values, np.r_[labels, rng.integers(0, 80, n)]


class Yolo11sNmsTests(unittest.TestCase):
    def test_identical_selection_and_order(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            boxes, values, labels = sample(rng)
            np.testing.assert_array_equal(nms_python(boxes, values, labels, .1, .5),
                                          nms_matrix(boxes, values, labels, .1, .5))

    def test_exact_iou_ties_on_quantized_grid(self):
        # W8A8 outputs sit on a coarse grid, so IoU == 0.5 exactly happens; keep the
        # baseline float32 decision instead of a float64 one.
        rng = np.random.default_rng(11)
        for _ in range(300):
            xy = rng.integers(0, 40, (300, 2)) * 4.0
            wh = rng.integers(2, 12, (300, 2)) * 4.0
            boxes = np.c_[xy, xy + wh].astype(np.float32)
            values = (rng.integers(1, 20, 300) / 20).astype(np.float32)
            labels = rng.integers(0, 2, 300)
            np.testing.assert_array_equal(nms_python(boxes, values, labels, .1, .5),
                                          nms_matrix(boxes, values, labels, .1, .5))

    def test_limit_and_empty(self):
        boxes, values, labels = sample(np.random.default_rng(3))
        self.assertEqual(len(nms_matrix(boxes, np.full_like(values, .9), np.zeros_like(labels), .1, 1.0)), 300)
        self.assertEqual(len(nms_matrix(boxes, np.zeros_like(values), labels, .1, .5)), 0)


if __name__ == "__main__":
    unittest.main()
