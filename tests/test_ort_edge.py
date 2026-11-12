"""GPU Edge path: the NPU adapters run their source ONNX on ORT CUDA, never a CPU fallback."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# Import before patching sys.modules: patch.dict would drop modules first imported inside it.
import cv2  # noqa: F401
import legacy.adapters.yolo.yolo_snpe  # noqa: F401
from inference import adapters, ort_session, yolo11s_snpe  # noqa: F401


def fake_ort(providers, outputs):
    def session(path, providers=None):
        s = mock.Mock()
        s.get_providers.return_value = list(providers_out)
        s.get_outputs.return_value = [types.SimpleNamespace(name=n) for n in outputs]
        s.get_inputs.return_value = [types.SimpleNamespace(name="images")]
        s.run.side_effect = lambda names, feed: [outputs[n](feed["images"]) for n in names]
        return s
    providers_out = providers
    return types.SimpleNamespace(InferenceSession=session, preload_dlls=lambda: None)


def v8_outputs():
    def boxes(x):
        assert x.shape == (1, 3, 320, 320)
        b = np.zeros((1, 4, 2100), np.float32)
        b[0, :, 0] = (160, 90, 64, 32)  # cx, cy, w, h in 320 letterbox space
        return b
    def scores(x):
        s = np.zeros((1, 80, 2100), np.float32)
        s[0, 2, 0] = .9  # one car
        return s
    return {"boxes": boxes, "scores": scores}


class OrtEdgeTests(unittest.TestCase):
    def test_refuses_cpu_fallback_and_wrong_outputs(self):
        from inference.ort_session import OrtCudaSession
        with mock.patch.dict(sys.modules, onnxruntime=fake_ort(["CPUExecutionProvider"], v8_outputs())):
            with self.assertRaises(RuntimeError):
                OrtCudaSession("m.onnx")
        with mock.patch.dict(sys.modules, onnxruntime=fake_ort(["CUDAExecutionProvider"], {"output0": None})):
            with self.assertRaises(ValueError):
                OrtCudaSession("m.onnx")

    def test_yolov8_onnx_uses_npu_preprocess_and_decode(self):
        from inference.adapters import YoloSnpeAdapter
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, onnxruntime=fake_ort(["CUDAExecutionProvider"], v8_outputs())):
            path = Path(tmp) / "yolov8n_320_split.onnx"
            path.write_bytes(b"x")
            adapter = YoloSnpeAdapter(dlc_path=str(path), imgsz=320, input_layout="nchw", score_threshold=.25)
            boxes, _ = adapter.infer(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(adapter.model_id, "yolov8n_320_split")
        self.assertEqual(adapter.detector.session.mode, "ort_cuda")
        (box,) = boxes
        self.assertEqual(box.class_id, 2)
        self.assertAlmostEqual(box.left, (160 - 32) / .25 / 1280, places=5)

    def test_yolo11s_onnx_is_pinned_and_nchw(self):
        from inference import yolo11s_snpe
        from yolo11s_contract import SHAPES, digest
        outs = {n: (lambda shape: lambda x: np.zeros(shape, np.float32))(s) for n, s in SHAPES.items()}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, onnxruntime=fake_ort(["CUDAExecutionProvider"], outs)):
            path = Path(tmp) / "yolo11s320_float.onnx"
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                yolo11s_snpe.Yolo11sSnpeAdapter(path, allow_diagnostic=True)
            with mock.patch.object(yolo11s_snpe, "ONNX_SHA256", digest(path)):
                adapter = yolo11s_snpe.Yolo11sSnpeAdapter(path, allow_diagnostic=True)
            prepared = adapter.prepare_frame(np.zeros((720, 1280, 3), np.uint8))
            boxes, _, _ = adapter.infer_prepared(prepared)
        self.assertEqual(prepared.tensor.shape, (1, 3, 320, 320))
        self.assertEqual((adapter.model_id, adapter.settings["session_mode"]), ("yolo11s320_float_ort", "ort_cuda"))
        self.assertEqual(boxes, [])


if __name__ == "__main__":
    unittest.main()
