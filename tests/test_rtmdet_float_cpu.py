import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from inference.model_artifact import load_artifact, sha256
from inference.rtmdet_float_cpu import RtmdetSmallFloatCpuAdapter


class FloatCpuTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "rtmdet_small_float.onnx"
        self.path.write_bytes(b"mock export")
        self.manifest = {
            "schema_version": 1, "kind": "rtmdet_small", "model_id": "rtmdet_s_8xb32_300e_coco",
            "sha256": sha256(self.path), "input_size": 640, "input_layout": "nchw",
            "output_shapes": {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]},
            "source_revision": "test", "checkpoint_sha256": "test", "license": "Apache-2.0",
            "label_map": "coco80_contiguous", "export_evidence": {
                "preprocess_vs_mmdetection": "PASS", "torch_onnx_raw_parity": "PASS",
                "onnx_decoder_vs_mmdetection": {"gate": "PASS"}, "nms_iou": .65,
                "input_mean": [103.53, 116.28, 123.675], "input_std": [57.375, 57.12, 58.395]},
        }
        self.write_manifest()
        self.session = MagicMock()
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        self.session.get_inputs.return_value = [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 640, 640])]
        self.session.get_outputs.return_value = [SimpleNamespace(name=name, type="tensor(float)", shape=shape)
                                                 for name, shape in self.manifest["output_shapes"].items()]
        self.boxes = np.zeros((1, 8400, 4), dtype=np.float32)
        self.scores = np.zeros((1, 8400, 80), dtype=np.float32)
        self.session.run.return_value = [self.boxes, self.scores]
        self.options = SimpleNamespace()
        self.ort = SimpleNamespace(SessionOptions=lambda: self.options,
                                   ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                                   InferenceSession=MagicMock(return_value=self.session))

    def write_manifest(self):
        Path(str(self.path) + ".json").write_text(json.dumps(self.manifest))

    def adapter(self, **kwargs):
        with patch.dict("sys.modules", {"onnxruntime": self.ort}):
            return RtmdetSmallFloatCpuAdapter(str(self.path), **kwargs)

    def test_onnx_requires_explicit_float_opt_in(self):
        with self.assertRaisesRegex(ValueError, "requires .dlc"):
            load_artifact(str(self.path), "rtmdet_small")
        self.assertEqual(load_artifact(str(self.path), "rtmdet_small", rtmdet_float_onnx=True)[0], self.path)
        with self.assertRaisesRegex(ValueError, "only to RTMDet"):
            load_artifact(str(self.path), "rtdetrv2_small", rtmdet_float_onnx=True)

    def test_cpu_only_and_bounded_threads(self):
        adapter = self.adapter(threads=2)
        self.assertEqual(adapter.runtime, "onnx_cpu")
        self.assertEqual(adapter.settings["precision"], "float32")
        self.assertEqual(self.options.intra_op_num_threads, 2)
        self.assertEqual(self.options.inter_op_num_threads, 1)
        self.assertEqual(self.options.execution_mode, "sequential")
        self.assertEqual(self.ort.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        for value in (0, 17, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.adapter(threads=value)

    def test_wrong_runtime_or_io_fails(self):
        self.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with self.assertRaises(RuntimeError):
            self.adapter()
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        self.session.get_inputs.return_value[0].shape = [1, 640, 640, 3]
        with self.assertRaises(ValueError):
            self.adapter()

    def test_unverified_contract_or_checksum_fails(self):
        self.manifest["export_evidence"]["preprocess_vs_mmdetection"] = "NOT_RUN"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "verified RTMDet"):
            self.adapter()
        self.path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.adapter()

    def test_nonsquare_bgr_padding_and_owned_tensor(self):
        adapter = self.adapter()
        image = np.full((180, 320, 3), [10, 30, 200], dtype=np.uint8)
        prepared = adapter.prepare_frame(image)
        expected = np.full((640, 640, 3), 114, dtype=np.float32)
        expected[:360] = cv2.resize(image, (640, 360))
        expected = (expected - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
        np.testing.assert_array_equal(prepared.tensor, expected.transpose(2, 0, 1)[None])
        self.assertEqual((prepared.scale_x, prepared.scale_y), (2, 2))
        self.assertTrue(prepared.tensor.flags.c_contiguous)
        before = prepared.tensor.copy()
        image[:] = 255
        np.testing.assert_array_equal(prepared.tensor, before)

    def test_clip_to_pinned_padded_shape_before_nms(self):
        adapter = self.adapter()
        self.boxes[0, :2] = [[0, 300, 100, 900], [0, 310, 100, 640]]
        self.scores[0, :2, 2] = [.9, .8]
        prepared = adapter.prepare_frame(np.zeros((360, 640, 3), np.uint8))
        boxes, elapsed, stages = adapter.infer_prepared(prepared)
        self.assertEqual(len(boxes), 1)  # overlap after 640-padding clipping exceeds .65
        self.assertEqual(boxes[0].class_name, "car")
        self.assertAlmostEqual(boxes[0].top, 300 / 360)
        self.assertEqual(boxes[0].bottom, 1)
        self.assertEqual(set(stages), {"edge_preprocess_ms", "edge_session_call_ms", "edge_postprocess_ms"})
        self.assertEqual(elapsed, sum(stages.values()))
        self.assertEqual(self.boxes[0, 0, 3], 900)  # session output is never mutated

    def test_invalid_input_and_output_rejected(self):
        adapter = self.adapter()
        for image in (None, np.zeros((0, 3, 3), np.uint8), np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.float32)):
            with self.assertRaises(ValueError):
                adapter.prepare_frame(image)
        prepared = adapter.prepare_frame(np.zeros((2, 2, 3), np.uint8))
        for boxes in (np.zeros((1, 8400, 4), np.float64), np.full((1, 8400, 4), np.nan, np.float32), np.zeros((1, 4), np.float32)):
            self.session.run.return_value = [boxes, self.scores]
            with self.assertRaises(ValueError):
                adapter.infer_prepared(prepared)


if __name__ == "__main__":
    unittest.main()
