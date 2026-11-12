import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference.adapters import YoloSnpeAdapter
from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
from scripts.test_live_yolo_infer_ab import infer_path, legacy_infer, legacy_preprocess, supervise


class LiveYoloInferAbTests(unittest.TestCase):
    def test_legacy_path_uses_detect_boxes_and_normalizes(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        seen = []

        class Detector:
            def detect_boxes(self, frame):
                seen.append(frame)
                return [dict(left=20, top=10, right=100, bottom=50,
                             confidence=.8, class_id=2, label="car"),
                        dict(left=10, top=10, right=10, bottom=50,
                             confidence=.8, class_id=2)]

        boxes, milliseconds = legacy_infer(SimpleNamespace(detector=Detector()), image)
        self.assertIs(seen[0], image)
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom),
                         (.1, .1, .5, .5))
        self.assertEqual(boxes[0].class_name, "car")
        self.assertGreaterEqual(milliseconds, 0)

    def test_invalid_images_are_rejected_before_detector(self):
        for image in (None, np.zeros((10, 10)), np.zeros((10, 10, 4))):
            with self.subTest(image_type=type(image)), self.assertRaises(ValueError):
                legacy_infer(SimpleNamespace(), image)

    def test_only_worker_infer_is_replaced_and_restored_on_error(self):
        original = YoloSnpeAdapter.infer
        prepare = YoloSnpeAdapter.prepare_frame
        preprocess = YOLOSnpeDetector._preprocess
        with self.assertRaises(RuntimeError):
            with infer_path("old-infer"):
                self.assertIs(YoloSnpeAdapter.infer, legacy_infer)
                self.assertIs(YoloSnpeAdapter.prepare_frame, prepare)
                self.assertIs(YOLOSnpeDetector._preprocess, preprocess)
                raise RuntimeError("test worker failure")
        self.assertIs(YoloSnpeAdapter.infer, original)
        with infer_path("current"):
            self.assertIs(YoloSnpeAdapter.infer, original)

    def test_old_full_restores_both_methods_on_failure(self):
        original = YoloSnpeAdapter.infer
        preprocess = YOLOSnpeDetector._preprocess
        with self.assertRaises(RuntimeError):
            with infer_path("old-full"):
                self.assertIs(YoloSnpeAdapter.infer, legacy_infer)
                self.assertIs(YOLOSnpeDetector._preprocess, legacy_preprocess)
                raise RuntimeError("worker failure")
        self.assertIs(YoloSnpeAdapter.infer, original)
        self.assertIs(YOLOSnpeDetector._preprocess, preprocess)

    def test_legacy_preprocess_matches_pixels_for_both_layouts(self):
        import cv2
        image = np.random.default_rng(17).integers(0, 256, (180, 320, 3), dtype=np.uint8)
        for layout in ("nchw", "nhwc"):
            detector = YOLOSnpeDetector.__new__(YOLOSnpeDetector)
            detector.imgsz, detector.input_layout = 320, layout
            detector._pad = detector._tensor = None
            current_tensor, current_ratio = detector._preprocess(image)
            current_tensor = current_tensor.copy()
            with patch.object(cv2, "cvtColor", wraps=cv2.cvtColor) as convert:
                old_tensor, old_ratio = legacy_preprocess(detector, image)
            self.assertIs(convert.call_args.args[0], image)
            np.testing.assert_array_equal(old_tensor, current_tensor)
            self.assertEqual(old_ratio, current_ratio)

    def test_watchdog_launches_same_case_and_reports_failure(self):
        argv = ["--ab-path", "old-infer", "--video", "clip.mp4"]
        with patch("scripts.test_live_yolo_infer_ab.subprocess.run",
                   return_value=SimpleNamespace(returncode=7)) as run:
            self.assertEqual(supervise(argv, "/source", 60), 7)
            command = run.call_args.args[0]
            self.assertIn("--ab-worker", command)
            self.assertEqual(command[-len(argv):], argv)
            self.assertEqual(run.call_args.kwargs, dict(cwd="/source", timeout=60))
        with patch("scripts.test_live_yolo_infer_ab.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("worker", 60)):
            self.assertEqual(supervise(argv, "/source", 60), 124)
