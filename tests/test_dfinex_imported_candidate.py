"""D-FINE-X (v50 accurate Cloud model): imported manifest, RT-DETRv2 IO contract, catalog wiring."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from contracts import ModelMode
from inference.catalog import phase7_cloud_catalog
from inference.cloud_detr import DfineXAdapter
from inference.model_artifact import candidate_model_id, load_artifact, sha256


class DfineXImportedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.artifact = Path(self.temp.name) / "dfine_x_obj2coco.onnx"
        self.artifact.write_bytes(b"fixture, not ONNX")
        digest = sha256(self.artifact)
        self.manifest = {"schema_version": 2, "manifest_type": "imported_candidate", "kind": "dfine_x",
                         "artifact": self.artifact.name, "model_id": candidate_model_id("dfine_x", digest),
                         "sha256": digest, "size_bytes": self.artifact.stat().st_size, "external_data": [],
                         "input_size": 640, "input_layout": "nchw",
                         "output_shapes": {"labels": [1, 300], "boxes": [1, 300, 4], "scores": [1, 300]},
                         "admission": {"production_ready": False}}
        Path(str(self.artifact) + ".json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def session(self, feeds):
        def run(names, feed):
            feeds.append(feed)
            labels = np.full((1, 300), 2, dtype=np.int64)  # contiguous car
            boxes = np.tile(np.array([128, 144, 640, 360], dtype=np.float32), (1, 300, 1))
            scores = np.zeros((1, 300), dtype=np.float32)
            scores[0, :3] = (.9, .7, .5)
            return [labels, boxes, scores]
        return SimpleNamespace(
            disable_fallback=lambda: None, get_providers=lambda: ["CUDAExecutionProvider"],
            get_inputs=lambda: [SimpleNamespace(name="images", type="tensor(float)", shape=["N", 3, 640, 640]),
                                SimpleNamespace(name="orig_target_sizes", type="tensor(int64)", shape=["N", 2])],
            get_outputs=lambda: [SimpleNamespace(name=n, shape=["N", 300, 4] if n == "boxes" else ["N", 300],
                                                 type="tensor(int64)" if n == "labels" else "tensor(float)")
                                 for n in ("labels", "boxes", "scores")],
            run=run)

    def test_manifest_requires_opt_in_and_self_contained_onnx(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            load_artifact(str(self.artifact), "dfine_x")
        with patch("inference.model_artifact._check_external_references"):
            _, data = load_artifact(str(self.artifact), "dfine_x", allow_imported_candidate=True)
        self.assertTrue(data["model_id"].startswith("dfine_x_candidate_"))
        sidecar = self.artifact.parent / "w.data"
        sidecar.write_bytes(b"x")
        self.manifest["external_data"] = [{"path": "w.data", "sha256": sha256(sidecar), "size_bytes": 1}]
        Path(str(self.artifact) + ".json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "self-contained"), \
                patch("inference.model_artifact._check_external_references"):
            load_artifact(str(self.artifact), "dfine_x", allow_imported_candidate=True)

    def test_adapter_feeds_plain_255_tensor_and_original_size(self):
        feeds = []
        with patch("inference.model_artifact._check_external_references"), \
                patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), \
                patch("onnxruntime.InferenceSession", return_value=self.session(feeds)):
            adapter = DfineXAdapter(str(self.artifact), confidence=.65, allow_imported_candidate=True)
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        boxes, _ = adapter.infer(frame)
        self.assertEqual(set(feeds[0]), {"images", "orig_target_sizes"})
        self.assertAlmostEqual(float(feeds[0]["images"].max()), 1.0, places=5)  # no ImageNet mean/std
        self.assertEqual(feeds[0]["orig_target_sizes"].tolist(), [[1280, 720]])
        self.assertEqual([round(b.confidence, 2) for b in boxes], [.9, .7])  # .5 is below the threshold
        self.assertEqual((boxes[0].class_name, round(boxes[0].left, 3), round(boxes[0].bottom, 3)), ("car", .1, .5))

    def test_catalog_registers_dfinex_mode_with_its_own_opt_in(self):
        with patch.dict(os.environ, {"DFINEX_ALLOW_IMPORTED_CANDIDATE": "1", "DFINEX_CONF_THR": ".65"}), \
                patch("inference.cloud_detr.DfineXAdapter",
                      return_value=SimpleNamespace(model_id="dx", infer=lambda x: ([], 0))) as factory:
            catalog = phase7_cloud_catalog(enabled=("rtdetrv2_small", "dfine_x"), rtdetrv2_onnx="rt.onnx",
                                           dfinex_onnx="dx.onnx")
            self.assertEqual(set(catalog.modes()), {ModelMode.RTDETR_CLOUD, ModelMode.DFINEX_CLOUD})
            catalog.load(ModelMode.DFINEX_CLOUD)
        self.assertEqual(factory.call_args.args, ("dx.onnx",))
        self.assertTrue(factory.call_args.kwargs["allow_imported_candidate"])
        self.assertEqual(factory.call_args.kwargs["confidence"], .65)
        with self.assertRaisesRegex(ValueError, "DFINEX_ONNX"):
            phase7_cloud_catalog(enabled=("dfine_x",))


if __name__ == "__main__":
    unittest.main()
