"""Imported bundle admission and runtime guards, never fake hardware acceptance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from inference.model_artifact import load_artifact, sha256, _check_external_references
from inference.cloud_detr import RtdetrV2SmallAdapter
from inference.catalog import phase7_cloud_catalog
from contracts import ModelMode


class ImportedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.artifact = Path(self.temp.name) / "rtdetrv2_s.onnx"
        self.artifact.write_bytes(b"fixture, not a real ONNX graph")
        self.weights = Path(str(self.artifact) + ".data")
        self.weights.write_bytes(b"fixture external weights")
        digest = sha256(self.artifact)
        self.manifest = {
            "schema_version": 2, "manifest_type": "imported_candidate", "kind": "rtdetrv2_small",
            "model_id": f"rtdetrv2_s_candidate_{digest[:12]}", "artifact": self.artifact.name,
            "sha256": digest, "size_bytes": self.artifact.stat().st_size,
            "external_data": [{"path": self.weights.name, "sha256": sha256(self.weights),
                               "size_bytes": self.weights.stat().st_size}],
            "input_size": 640, "input_layout": "nchw",
            "output_shapes": {"labels": [1, 300], "boxes": [1, 300, 4], "scores": [1, 300]},
            "provenance": {"verified": False}, "admission": {"production_ready": False},
        }
        self.save()

    def save(self):
        Path(str(self.artifact) + ".json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def load(self):
        return load_artifact(str(self.artifact), "rtdetrv2_small", allow_imported_candidate=True)

    def test_opt_in_and_unknown_checkpoint_identity(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            load_artifact(str(self.artifact), "rtdetrv2_small")
        with patch("inference.model_artifact._check_external_references") as graph_check:
            _, manifest = self.load()
        graph_check.assert_called_once_with(self.artifact.resolve(), {self.weights.name})
        self.assertIn("candidate", manifest["model_id"])
        self.assertFalse(manifest["provenance"]["verified"])

    def test_modified_external_weights_rejected(self):
        self.weights.write_bytes(b"x" * self.weights.stat().st_size)
        with self.assertRaisesRegex(ValueError, "checksum"), patch("inference.model_artifact._check_external_references") as check:
            self.load()
        check.assert_not_called()

    def test_missing_external_weights_rejected(self):
        self.weights.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_external_escape_duplicates_and_wrong_inventory_rejected(self):
        original = self.manifest["external_data"][0].copy()
        for path in ("../weights.data", "C:\\weights.data", "/weights.data", "other.data"):
            with self.subTest(path=path):
                self.manifest["external_data"] = [{**original, "path": path}]
                self.save()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load()
        self.manifest["external_data"] = [original, original]
        self.save()
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load()

    def test_no_sp2_alias_or_production_ready_claim(self):
        original = self.manifest.copy()
        for update in ({"model_id": "rtdetrv2_r18vd_sp2_120e_coco"},
                       {"admission": {"production_ready": True}},
                       {"output_shapes": {"labels": [2, 300]}}):
            self.manifest = {**original, **update}
            self.save()
            with self.assertRaises(ValueError):
                self.load()

    def fake_session(self):
        return SimpleNamespace(
            disable_fallback=lambda: None, get_providers=lambda: ["CUDAExecutionProvider"],
            get_inputs=lambda: [SimpleNamespace(name="images", type="tensor(float)", shape=["N", 3, 640, 640]),
                                SimpleNamespace(name="orig_target_sizes", type="tensor(int64)", shape=["N", 2])],
            get_outputs=lambda: [SimpleNamespace(name=name, shape=shape, type="tensor(int64)" if name == "labels" else "tensor(float)")
                                 for name, shape in self.manifest["output_shapes"].items()],
            run=lambda names, feed: [np.full((1, 300), 2, dtype=np.int64),
                np.tile(np.array([128, 144, 640, 360], dtype=np.float32), (1, 300, 1)),
                np.full((1, 300), .8, dtype=np.float32)])

    def test_candidate_adapter_preserves_identity_and_enforces_types(self):
        session = self.fake_session()
        with patch("inference.model_artifact._check_external_references"), \
                patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), \
                patch("onnxruntime.InferenceSession", return_value=session):
            adapter = RtdetrV2SmallAdapter(str(self.artifact), allow_imported_candidate=True)
        self.assertEqual(adapter.model_id, self.manifest["model_id"])
        self.assertEqual(adapter.settings["label_map_status"], "assumed_unverified")
        boxes, _ = adapter.infer(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(len(boxes), 300)
        self.assertAlmostEqual(boxes[0].left, .1)
        session.run = lambda names, feed: [np.zeros((1, 300), dtype=np.float32),
            np.zeros((1, 300, 4), dtype=np.float32), np.ones((1, 300), dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, "dtype"):
            adapter.infer(np.zeros((720, 1280, 3), dtype=np.uint8))

    def test_candidate_never_allows_cpu_fallback(self):
        with patch("inference.model_artifact._check_external_references"), \
                patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                RtdetrV2SmallAdapter(str(self.artifact), allow_imported_candidate=True)

    def test_wrong_runtime_output_metadata_rejected_before_inference(self):
        session = self.fake_session()
        outputs = session.get_outputs()
        outputs[0].type = "tensor(float)"
        session.get_outputs = lambda: outputs
        with patch("inference.model_artifact._check_external_references"), \
                patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), \
                patch("onnxruntime.InferenceSession", return_value=session):
            with self.assertRaisesRegex(ValueError, "metadata"):
                RtdetrV2SmallAdapter(str(self.artifact), allow_imported_candidate=True)

    def test_catalog_opt_in_is_scoped_to_rtdetr(self):
        with patch.dict(os.environ, {"RTDETRV2_ALLOW_IMPORTED_CANDIDATE": "1"}), \
                patch("inference.cloud_detr.RtdetrV2SmallAdapter", return_value=SimpleNamespace(model_id="candidate", infer=lambda x: ([], 0))) as factory:
            phase7_cloud_catalog(enabled=("rtdetrv2_small",), rtdetrv2_onnx="model.onnx").load(ModelMode.RTDETR_CLOUD)
            self.assertTrue(factory.call_args.kwargs["allow_imported_candidate"])
        with patch.dict(os.environ, {"RTDETRV2_ALLOW_IMPORTED_CANDIDATE": "0"}), \
                patch("inference.cloud_detr.RtdetrV2SmallAdapter", return_value=SimpleNamespace(model_id="candidate", infer=lambda x: ([], 0))) as factory:
            phase7_cloud_catalog(enabled=("rtdetrv2_small",), rtdetrv2_onnx="model.onnx").load(ModelMode.RTDETR_CLOUD)
            self.assertFalse(factory.call_args.kwargs["allow_imported_candidate"])


class ExternalGraphTests(unittest.TestCase):
    def test_hidden_external_file_and_out_of_bounds_range_are_rejected(self):
        import onnx
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.data").write_bytes(b"1234")
            path = root / "model.onnx"
            for location, length in (("unhashed.data", "4"), ("weights.data", "5")):
                tensor = onnx.TensorProto(name="weight", data_type=onnx.TensorProto.FLOAT,
                                          dims=[1], data_location=onnx.TensorProto.EXTERNAL)
                for key, value in (("location", location), ("offset", "0"), ("length", length)):
                    tensor.external_data.add(key=key, value=value)
                model = onnx.helper.make_model(onnx.helper.make_graph([], "test", [], [], [tensor]))
                path.write_bytes(model.SerializeToString())
                with self.assertRaisesRegex(ValueError, "inventory|range"):
                    _check_external_references(path, {"weights.data"})
