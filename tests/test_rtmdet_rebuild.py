import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from contextlib import redirect_stderr
import io
from contracts import Box
from scripts.rebuild_rtmdet_small import compare_detections, load_strict_checkpoint, REVISION, CHECKPOINT_SHA256
from scripts.prepare_rtmdet_rebuild_calibration import checked_reference, calibration_tensor, runtime_paths, main as calibration_main


class RebuildParityTests(unittest.TestCase):
    def box(self, *, cls=2, left=.1, score=.8):
        return Box(left, .2, .5, .6, score, cls, "car")

    def test_identical_and_reordered(self):
        boxes = [self.box(), self.box(cls=0, left=.3)]
        self.assertEqual(compare_detections(boxes, boxes[::-1])["gate"], "PASS")

    def test_small_numerical_difference(self):
        result = compare_detections([self.box()], [self.box(left=.10001, score=.80001)])
        self.assertGreater(result["max_score_error"], 0)

    def test_reject_empty_wrong_count_class_box_and_score(self):
        for expected, actual in [([], []), ([self.box()], []),
                                 ([self.box()], [self.box(cls=3)]),
                                 ([self.box()], [self.box(left=.3)]),
                                 ([self.box()], [self.box(score=.9)])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                compare_detections(expected, actual)

    def test_duplicate_cannot_match_twice(self):
        with self.assertRaises(ValueError):
            compare_detections([self.box(), self.box()], [self.box(), self.box(left=.4)])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            compare_detections([self.box()], [self.box(score=float("nan"))])


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        import torch
        model = torch.nn.Module()
        model.data_preprocessor = torch.nn.Module()
        model.data_preprocessor.register_buffer("mean", torch.tensor([1., 2., 3.]), persistent=False)
        model.data_preprocessor.register_buffer("std", torch.tensor([4., 5., 6.]), persistent=False)
        model.layer = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["data_preprocessor.mean"] = model.data_preprocessor.mean.clone()
        state["data_preprocessor.std"] = model.data_preprocessor.std.clone()
        return model, state

    def test_verified_old_buffers_only(self):
        model, state = self.fixture()
        self.assertEqual(load_strict_checkpoint(model, state), ["data_preprocessor.mean", "data_preprocessor.std"])
        self.assertIn("data_preprocessor.mean", state)  # caller state untouched

    def test_wrong_constant_missing_weight_and_unknown_key_rejected(self):
        for change in ("constant", "missing_weight", "unknown"):
            model, state = self.fixture()
            if change == "constant":
                state["data_preprocessor.mean"][0] = 99
            elif change == "missing_weight":
                state.pop("layer.weight")
            else:
                state["other_parameter"] = state["layer.weight"]
            with self.subTest(change=change), self.assertRaises((RuntimeError, AssertionError)):
                load_strict_checkpoint(model, state)


class CalibrationGateTests(unittest.TestCase):
    def test_nhwc_requires_explicit_dlc_io_gate_before_file_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "calibration"
            argv = ["calibration", "--reference-dir", tmp, "--video", "unused.mp4",
                    "--output-dir", str(target), "--layout", "nhwc"]
            with patch("sys.argv", argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                calibration_main()
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse(target.exists())

    def test_layout_transpose_exact_no_arithmetic(self):
        import numpy as np
        source = np.arange(3 * 640 * 640, dtype=np.float32).reshape(1, 3, 640, 640)
        target = calibration_tensor(source, "nhwc")
        self.assertEqual(target.shape, (1, 640, 640, 3))
        self.assertTrue(target.flags.c_contiguous)
        np.testing.assert_array_equal(target.transpose(0, 3, 1, 2), source)
        np.testing.assert_array_equal(calibration_tensor(source, "nchw"), source)

    def test_layout_reject_wrong_shape_dtype_nonfinite_and_layout(self):
        import numpy as np
        source = np.zeros((1, 3, 640, 640), dtype=np.float32)
        nonfinite = source.copy()
        nonfinite[0, 0, 0, 0] = float("nan")
        for tensor, layout in [(source[:, :, :1], "nhwc"), (source.astype(np.uint8), "nhwc"),
                               (nonfinite, "nhwc"), (source, "unknown")]:
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                calibration_tensor(tensor, layout)

    def test_runtime_paths_linux_absolute_and_reject_traversal(self):
        self.assertEqual(runtime_paths("/home/stu0/calibration", ["frame_000000.raw"]),
                         ["/home/stu0/calibration/frame_000000.raw"])
        for root in ("", "relative", "C:\\samples", "/a/../b", "/a b", "/a\nb", "//a", "/a/"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                runtime_paths(root, ["frame.raw"])
        for name in ("../frame.raw", "/frame.raw", "a\\frame.raw", "a b.raw", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime_paths("/home/stu0/calibration", [name])

    def test_pass_and_reject_tampered_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            artifact = folder / "rtmdet_small_float.onnx"
            artifact.write_bytes(b"reference fixture")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            evidence = {"preprocess_vs_mmdetection": "PASS", "torch_onnx_raw_parity": "PASS",
                        "onnx_decoder_vs_mmdetection": {"gate": "PASS"}}
            report = {"export_gate": "PASS", "source_revision": REVISION,
                      "checkpoint_sha256": CHECKPOINT_SHA256, "artifact_sha256": digest, "evidence": evidence}
            manifest = {"sha256": digest, "input_layout": "nchw", "input_size": 640,
                        "source_revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                        "output_shapes": {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]},
                        "export_evidence": evidence}
            (folder / "rebuild_report.json").write_text(json.dumps(report), encoding="utf-8")
            Path(str(artifact) + ".json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(checked_reference(folder)[0], artifact)
            artifact.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                checked_reference(folder)


if __name__ == "__main__":
    unittest.main()
