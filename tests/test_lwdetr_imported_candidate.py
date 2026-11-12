"""Imported LW-DETR admission, CUDA guards and source-coordinate smoke output."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts import Box, ModelMode
from inference.catalog import phase7_cloud_catalog
from inference.cloud_detr import LwDetrLargeAdapter
from inference.model_artifact import candidate_model_id, load_artifact, sha256
from scripts.candidate_artifact import verify


class LwImportedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "lwdetr_large_coco.onnx"
        self.artifact.write_bytes(b"fixture, not ONNX")
        digest = sha256(self.artifact)
        self.data = {"schema_version": 2, "manifest_type": "imported_candidate", "kind": "lwdetr_large",
                     "artifact": self.artifact.name, "model_id": candidate_model_id("lwdetr_large", digest),
                     "sha256": digest, "size_bytes": self.artifact.stat().st_size, "external_data": [],
                     "input_size": 640, "input_layout": "nchw",
                     "output_shapes": {"dets": [1, 300, 4], "labels": [1, 300, 91]},
                     "admission": {"production_ready": False}}
        self.save()

    def save(self):
        Path(str(self.artifact) + ".json").write_text(json.dumps(self.data), encoding="utf-8")

    def load(self):
        return load_artifact(str(self.artifact), "lwdetr_large", allow_imported_candidate=True)

    def session(self):
        coords = np.tile(np.array([.5, .5, .5, .5], dtype=np.float32), (1, 300, 1))
        logits = np.full((1, 300, 91), -80, dtype=np.float32)
        logits[0, 0, 3] = 4  # COCO category car=3 -> contiguous class2.
        return SimpleNamespace(
            disable_fallback=lambda: None, get_providers=lambda: ["CUDAExecutionProvider"],
            get_inputs=lambda: [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 640, 640])],
            get_outputs=lambda: [SimpleNamespace(name="dets", type="tensor(float)", shape=["B", "Q", 4]),
                                 SimpleNamespace(name="labels", type="tensor(float)", shape=["B", "Q", 91])],
            run=lambda names, feed: [coords, logits])

    def adapter(self, session):
        with patch("inference.model_artifact._check_external_references"), \
                patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), \
                patch("onnxruntime.InferenceSession", return_value=session):
            return LwDetrLargeAdapter(str(self.artifact), allow_imported_candidate=True)

    def test_default_off_self_contained_and_hash_bound_identity(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            load_artifact(str(self.artifact), "lwdetr_large")
        with patch("inference.model_artifact._check_external_references") as check:
            _, data = self.load()
        check.assert_called_once_with(self.artifact.resolve(), set())
        self.assertEqual(data["model_id"], self.data["model_id"])
        self.data["model_id"] = "lwdetr_large_coco"
        self.save()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.load()

    def test_modified_onnx_rejected(self):
        self.artifact.write_bytes(b"f" * self.artifact.stat().st_size)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load()

    def test_external_inventory_and_production_claim_rejected(self):
        for update in ({"external_data": None}, {"admission": {"production_ready": True}},
                       {"output_shapes": {"dets": [1, 300, 4], "labels": [1, 300, 80]}}):
            before = self.data.copy()
            self.data.update(update)
            self.save()
            with self.assertRaises(ValueError):
                self.load()
            self.data = before

    def test_symbolic_metadata_allowed_actual_outputs_still_strict(self):
        session = self.session()
        adapter = self.adapter(session)
        boxes, _ = adapter.infer(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertEqual(adapter.model_id, self.data["model_id"])
        self.assertEqual([(b.class_id, b.class_name) for b in boxes], [(2, "car")])
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom), (.25, .25, .75, .75))
        session.run = lambda names, feed: [np.zeros((1, 299, 4), dtype=np.float32),
                                          np.zeros((1, 299, 91), dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.infer(np.zeros((360, 640, 3), dtype=np.uint8))
        session.run = lambda names, feed: [np.zeros((1, 300, 4), dtype=np.float64),
                                          np.zeros((1, 300, 91), dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, "dtype"):
            adapter.infer(np.zeros((360, 640, 3), dtype=np.uint8))

    def test_wrong_concrete_metadata_and_logits_type_rejected(self):
        for shape, dtype in (([1, 300, 80], "tensor(float)"), (["B", "Q", 91], "tensor(int64)")):
            session = self.session()
            outputs = session.get_outputs()
            outputs[1].shape, outputs[1].type = shape, dtype
            session.get_outputs = lambda: outputs
            with self.assertRaisesRegex(ValueError, "metadata"):
                self.adapter(session)

    def test_cpu_only_provider_and_cuda_initialization_fallback_rejected(self):
        with patch("inference.model_artifact._check_external_references"), \
                patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                LwDetrLargeAdapter(str(self.artifact), allow_imported_candidate=True)
        session = self.session()
        session.get_providers = lambda: ["CPUExecutionProvider"]
        with self.assertRaisesRegex(RuntimeError, "initialization"):
            self.adapter(session)

    def test_candidate_opt_in_environment_is_model_scoped(self):
        for lw, rt in (("0", "1"), ("1", "0")):
            with patch.dict(os.environ, {"LWDETR_ALLOW_IMPORTED_CANDIDATE": lw,
                                         "RTDETRV2_ALLOW_IMPORTED_CANDIDATE": rt}), \
                    patch("inference.cloud_detr.LwDetrLargeAdapter", return_value=SimpleNamespace(
                        model_id="candidate", infer=lambda x: ([], 0))) as factory:
                catalog = phase7_cloud_catalog(enabled=("lwdetr_large",), lwdetr_onnx="model.onnx")
                catalog.load(ModelMode.LWDETR_CLOUD)
                self.assertEqual(factory.call_args.kwargs["allow_imported_candidate"], lw == "1")

    def test_video_smoke_creates_original_size_overlay_and_cuda_report(self):
        from PIL import Image
        pixels = np.zeros((360, 640, 3), dtype=np.uint8)
        profile = self.root / "profile.json"
        profile.write_text(json.dumps([{"cat": "Node", "args": {
            "provider": "CUDAExecutionProvider", "op_name": "Conv"}}]))
        session = SimpleNamespace(end_profiling=lambda: str(profile))
        adapter = SimpleNamespace(model_id=self.data["model_id"], runtime="cuda", settings={}, session=session,
            infer=lambda image: ([Box(.25, .25, .75, .75, .9, 2, "car")], 1.0))
        capture = SimpleNamespace(read=lambda: (True, pixels), release=lambda: None)
        report, overlay = self.root / "report.json", self.root / "overlay.png"
        with patch("cv2.VideoCapture", return_value=capture), \
                patch("inference.cloud_detr.LwDetrLargeAdapter", return_value=adapter), \
                contextlib.redirect_stdout(io.StringIO()):
            status = verify(self.artifact, "lwdetr_large", None, report, 1,
                            allow_imported_candidate=True, video=self.root / "video.mp4", overlay_path=overlay)
        self.assertEqual(status, 0)
        data = json.loads(report.read_text())
        self.assertTrue(data["cuda_active"])
        self.assertEqual(data["quality_status"], "NOT_EVALUATED")
        self.assertEqual(data["last_boxes"][0]["class_id"], 2)
        with Image.open(overlay) as image:
            self.assertEqual(image.size, (640, 360))
            self.assertEqual(image.getpixel((320, 270)), (0, 255, 0))
        with self.assertRaises(FileExistsError):
            verify(self.artifact, "lwdetr_large", None, report, 1, video=self.root / "video.mp4")


if __name__ == "__main__":
    unittest.main()
