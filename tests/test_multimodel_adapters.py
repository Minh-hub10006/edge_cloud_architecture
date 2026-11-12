from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from inference.cloud_detr import CloudDetrAdapter, RtdetrV2SmallAdapter, LwDetrLargeAdapter
from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter
from inference.detection_decode import rtdetr_decode, lwdetr_decode, rtmdet_decode, COCO_NAMES, COCO_CATEGORY_IDS
from inference.model_artifact import MODEL_IDS, load_artifact, sha256
from inference.catalog import phase7_cloud_catalog, phase7_edge_catalog
from contracts import ModelMode


class DecoderTests(unittest.TestCase):
    def test_coco_mapping_is_unique_and_complete(self):
        self.assertEqual(len(COCO_NAMES), 80)
        self.assertEqual(len(set(COCO_CATEGORY_IDS)), 80)
        self.assertEqual(COCO_NAMES[2], "car")

    def test_rtdetr_nonsquare_original_size_pixel_boxes(self):
        outputs = {"boxes": np.array([[[128, 144, 640, 360], [0, 0, 1, 1]]]),
                   "labels": np.array([[2, 0]]), "scores": np.array([[.8, .01]])}
        boxes = rtdetr_decode(outputs, 1280, 720, .1, 300)
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom), (.1, .2, .5, .5))
        self.assertEqual(boxes[0].class_name, "car")

    def test_rtdetr_invalid_dimensions_probabilities_labels_nan(self):
        good = {"boxes": np.array([[[1., 2., 3., 4.]]]), "labels": np.array([[2.]]), "scores": np.array([[.8]])}
        for key, invalid in (("labels", np.array([[90]])), ("labels", np.array([[2.5]])),
                             ("scores", np.array([[2.]])), ("boxes", np.full((1, 1, 4), np.nan)),
                             ("boxes", np.zeros((1, 4)))):
            with self.subTest(key=key, invalid=invalid.shape), self.assertRaises(ValueError):
                rtdetr_decode({**good, key: invalid}, 640, 480, .1, 300)

    def test_lwdetr_final_decoder_layer_and_multilabel_topk(self):
        coords = np.full((3, 1, 1, 4), .2)
        coords[-1] = [.5, .5, .4, .2]
        logits = np.full((3, 1, 1, 91), -80.)
        logits[0, 0, 0, 6] = 80.  # earlier decoder should never be used
        logits[-1, 0, 0, 3] = 4.  # sparse car -> contiguous 2
        logits[-1, 0, 0, 1] = 3.  # same query, another class: do NOT argmax
        boxes = lwdetr_decode({"dets": coords, "labels": logits}, .1, 300)
        self.assertEqual([b.class_id for b in boxes], [2, 0])
        self.assertAlmostEqual(boxes[0].left, .3)
        self.assertAlmostEqual(boxes[0].bottom, .6)

    def test_lwdetr_padding_categories_not_reported(self):
        logits = np.full((1, 1, 91), -80.)
        logits[0, 0, 12] = 80  # no COCO category 12
        logits[0, 0, 90] = 4
        boxes = lwdetr_decode({"dets": np.array([[[.5, .5, .4, .2]]]), "labels": logits}, .1, 300)
        self.assertEqual([b.class_name for b in boxes], ["toothbrush"])

    def test_lwdetr_rejects_contiguous80_export_and_batch2(self):
        for coords, logits in ((np.zeros((1, 1, 4)), np.zeros((1, 1, 80))),
                               (np.zeros((2, 1, 4)), np.zeros((2, 1, 91)))):
            with self.assertRaises(ValueError):
                lwdetr_decode({"dets": coords, "labels": logits}, .1, 300)

    def test_rtmdet_classwise_nms_and_padding_inverse(self):
        scores = np.zeros((1, 3, 80))
        scores[0, :, 2] = [.9, .8, .7]
        scores[0, 1, 0] = .85
        outputs = {"boxes": np.array([[[64, 36, 320, 180], [64, 36, 320, 180], [480, 10, 600, 40]]]), "scores": scores}
        boxes = rtmdet_decode(outputs, width=1280, height=720, scale_x=.5, scale_y=.5, confidence=.1, top_k=300)
        self.assertEqual([b.class_id for b in boxes], [2, 0, 2])
        self.assertEqual((boxes[0].left, boxes[0].top), (.1, .1))

    def test_rtmdet_all_filtered_and_logits_rejected(self):
        outputs = {"boxes": np.zeros((1, 2, 4)), "scores": np.zeros((1, 2, 80))}
        kwargs = dict(width=1280, height=720, scale_x=.5, scale_y=.5, confidence=.1, top_k=300)
        self.assertEqual(rtmdet_decode(outputs, **kwargs), [])
        outputs["scores"][0, 0, 0] = 2
        with self.assertRaises(ValueError):
            rtmdet_decode(outputs, **kwargs)


class ArtifactAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def artifact(self, kind="rtdetrv2_small"):
        path = self.root / (kind + (".dlc" if kind == "rtmdet_small" else ".onnx"))
        path.write_bytes(b"test export contract only")
        shapes = {"rtdetrv2_small": {"labels": [1, 1], "boxes": [1, 1, 4], "scores": [1, 1]},
                  "lwdetr_large": {"dets": [3, 1, 1, 4], "labels": [3, 1, 1, 91]},
                  "rtmdet_small": {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]}}[kind]
        manifest = {"schema_version": 1, "kind": kind, "model_id": MODEL_IDS[kind], "sha256": sha256(path),
                    "input_size": 640, "input_layout": "nhwc" if kind == "rtmdet_small" else "nchw",
                    "output_shapes": shapes, "source_revision": "test", "checkpoint_sha256": "test",
                    "license": "Apache-2.0", "label_map": "coco80_contiguous"}
        Path(str(path) + ".json").write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest

    def test_manifest_rejects_wrong_model_sha_layout_shapes_kind(self):
        path, manifest = self.artifact()
        for key, value in (("sha256", "bad"), ("model_id", "dfine"), ("kind", "lwdetr_large"),
                           ("input_layout", "nhwc"), ("input_size", 320), ("output_shapes", {"x": [1]}),
                           ("source_revision", ""), ("label_map", "coco91")):
            with self.subTest(key=key):
                Path(str(path) + ".json").write_text(json.dumps({**manifest, key: value}))
                with self.assertRaises(ValueError):
                    load_artifact(str(path), "rtdetrv2_small")

    def test_cloud_missing_cuda_fails_before_session_creation(self):
        path, _ = self.artifact()
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]), patch("onnxruntime.InferenceSession") as session:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                RtdetrV2SmallAdapter(str(path))
            session.assert_not_called()

    def test_cloud_cuda_init_fallback_rejected(self):
        path, _ = self.artifact()
        fake = SimpleNamespace(disable_fallback=lambda: None, get_providers=lambda: ["CPUExecutionProvider"])
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), patch("onnxruntime.InferenceSession", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "initialization"):
                RtdetrV2SmallAdapter(str(path))

    def test_cloud_preprocess_bgr_rgb_original_wh_and_lw_normalization(self):
        pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
        pixels[..., 2] = 255
        for mode in ("pil", "cv2"):
            adapter = object.__new__(CloudDetrAdapter)
            adapter.kind, adapter.preprocess = "rtdetrv2_small", mode
            feed = adapter.prepare(pixels)
            self.assertEqual(feed["orig_target_sizes"].tolist(), [[1280, 720]])
            self.assertEqual(feed["images"].shape, (1, 3, 640, 640))
            np.testing.assert_allclose(feed["images"][0, :, 0, 0], [1, 0, 0])
            adapter.kind = "lwdetr_large"
            np.testing.assert_allclose(adapter.prepare(pixels)["input"][0, :, 0, 0], [(1-.485)/.229, -.456/.224, -.406/.225], rtol=1e-6)

    def test_cloud_full_infer_fake_signature_and_manifest_shape_check(self):
        path, _ = self.artifact()
        fake = SimpleNamespace(disable_fallback=lambda: None, get_providers=lambda: ["CUDAExecutionProvider"],
            get_inputs=lambda: [SimpleNamespace(name="images", type="tensor(float)", shape=["N", 3, 640, 640]),
                                SimpleNamespace(name="orig_target_sizes", type="tensor(int64)", shape=["N", 2])],
            get_outputs=lambda: [SimpleNamespace(name=n) for n in ["labels", "boxes", "scores"]],
            run=lambda names, feed: [np.array([[2]]), np.array([[[128,144,640,360]]]), np.array([[.8]])])
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), patch("onnxruntime.InferenceSession", return_value=fake):
            adapter = RtdetrV2SmallAdapter(str(path))
            boxes, elapsed, stages = adapter.infer_detailed(np.zeros((720,1280,3), dtype=np.uint8))
            self.assertEqual(boxes[0].class_name, "car")
            self.assertGreater(elapsed, 0)
            self.assertIn("cloud_session_call_ms", stages)
            fake.run = lambda n, f: [np.array([[2,2]]), np.zeros((1,2,4)), np.ones((1,2))]
            with self.assertRaisesRegex(ValueError, "shape"):
                adapter.infer(np.zeros((720,1280,3), dtype=np.uint8))

    def test_rtmdet_owned_tensors_bgr_nhwc_padding_no_yolo(self):
        adapter = object.__new__(RtmdetSmallSnpeAdapter)
        adapter.manifest = {"input_layout": "nhwc"}
        pixels = np.zeros((720,1280,3), dtype=np.uint8)
        first = adapter.prepare_frame(pixels)
        second = adapter.prepare_frame(np.full_like(pixels, 255))
        self.assertEqual(first.tensor.shape, (1,640,640,3))
        self.assertFalse(np.shares_memory(first.tensor, second.tensor))
        self.assertEqual((first.scale_x, first.scale_y), (.5,.5))
        np.testing.assert_allclose(first.tensor[0,0,0], -np.array([103.53,116.28,123.675])/np.array([57.375,57.12,58.395]), rtol=1e-6)
        np.testing.assert_allclose(first.tensor[0,500,0], (114-np.array([103.53,116.28,123.675]))/np.array([57.375,57.12,58.395]), rtol=1e-5)

    def test_rtmdet_native_flat_outputs_decode(self):
        path, manifest = self.artifact("rtmdet_small")
        outputs = {"boxes": np.zeros((1,8400,4), dtype=np.float32), "scores": np.zeros((1,8400,80), dtype=np.float32)}
        outputs["boxes"][0,0] = [64,36,320,180]
        outputs["scores"][0,0,2] = .9
        session = SimpleNamespace(runtime="dsp", mode="snpe_daemon", infer_raw=lambda x: (.003, {n:v.flatten() for n,v in outputs.items()}))
        with patch("legacy.adapters.npu.snpe_dlc_session.SnpeDlcSession", return_value=session):
            adapter = RtmdetSmallSnpeAdapter(str(path))
            boxes, ms, stages = adapter.infer_prepared(adapter.prepare_frame(np.zeros((720,1280,3), dtype=np.uint8)))
            self.assertEqual(boxes[0].class_id, 2)
            self.assertEqual(stages["edge_npu_reported_ms"], 3)
            self.assertGreater(ms, 0)

    def test_rtmdet_rejects_subprocess_fallback(self):
        path, _ = self.artifact("rtmdet_small")
        with patch("legacy.adapters.npu.snpe_dlc_session.SnpeDlcSession", return_value=SimpleNamespace(runtime="dsp", mode="snpe_subprocess")):
            with self.assertRaisesRegex(RuntimeError, "persistent"):
                RtmdetSmallSnpeAdapter(str(path))

    def test_catalog_default_and_opt_in_and_invalid(self):
        self.assertEqual(phase7_cloud_catalog().modes(), (ModelMode.DFINE_CLOUD,))
        self.assertEqual(phase7_cloud_catalog(rtdetrv2_onnx="candidate.onnx", lwdetr_onnx="candidate2.onnx",
            enabled=("rtdetrv2_small", "lwdetr_large")).modes(), (ModelMode.RTDETR_CLOUD, ModelMode.LWDETR_CLOUD))
        self.assertIn(ModelMode.RTMDET_EDGE, phase7_edge_catalog(rtmdet_dlc="candidate.dlc").modes())
        for enabled in ((), ("nope",), ("dfine", "dfine"), ("rtdetrv2_small",)):
            with self.assertRaises(ValueError):
                phase7_cloud_catalog(enabled=enabled)
