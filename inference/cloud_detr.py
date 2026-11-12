"""CUDA ONNX adapters matching official RT-DETRv2/LW-DETR/D-FINE export contracts."""
from __future__ import annotations

import os
from time import perf_counter
import numpy as np

from .model_artifact import load_artifact, OUTPUT_NAMES
from .detection_decode import rtdetr_decode, lwdetr_decode

_IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_INV_STD = 1 / np.array([.229, .224, .225], dtype=np.float32).reshape(1, 3, 1, 1)
# D-FINE deploy keeps RT-DETRv2's IO: images /255 + orig_target_sizes -> contiguous labels, xyxy pixels, scores.
_RTDETR_IO = {"rtdetrv2_small", "dfine_x"}


class CloudDetrAdapter:
    def __init__(self, onnx_path: str, *, kind: str, manifest_path: str | None = None,
                 confidence: float = 0.10, top_k: int = 300,
                 allow_imported_candidate: bool = False):
        if not 0 <= confidence <= 1 or type(top_k) is not int or not 1 <= top_k <= 300:
            raise ValueError("confidence must be 0..1 and top_k 1..300")
        self.artifact_path, self.manifest = load_artifact(onnx_path, kind, manifest_path,
            allow_imported_candidate=allow_imported_candidate)
        self.imported_candidate = self.manifest.get("schema_version") == 2
        self.model_id, self.kind = self.manifest["model_id"], kind
        self.confidence, self.top_k = confidence, top_k
        # "pil" = baseline 2026-10-10 (16-18 ms/frame), kept for A/B and rollback.
        self.preprocess = os.getenv("DETR_PREPROCESS", "cv2")
        if self.preprocess not in {"cv2", "pil"}:
            raise ValueError("DETR_PREPROCESS must be cv2 or pil")
        import onnxruntime as ort
        # CLOUD_RUNTIME=triton (v4): the same sealed ONNX, served by Triton (inference/triton_detr.py); a CPU
        # session is kept only to check the export's IO contract below, it never runs.
        triton = os.getenv("CLOUD_RUNTIME", "onnxruntime") == "triton"
        if triton and kind not in _RTDETR_IO:
            raise ValueError(f"CLOUD_RUNTIME=triton serves rtdetrv2_small and dfine_x only, not {kind}")
        if not triton and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider required; CPU-only fallback is forbidden")
        options = ort.SessionOptions()
        options.enable_profiling = os.getenv("MODEL_ORT_PROFILE", "0") == "1"
        self.session = ort.InferenceSession(str(self.artifact_path), sess_options=options,
            providers=["CPUExecutionProvider"] if triton else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        if not triton:
            self.session.disable_fallback()
            if self.session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA session initialization failed")
        self.runtime = "cuda"
        inputs = {item.name: item for item in self.session.get_inputs()}
        image_name = "images" if kind in _RTDETR_IO else "input"
        expected = {image_name, "orig_target_sizes"} if kind in _RTDETR_IO else {image_name}
        if set(inputs) != expected or inputs[image_name].type != "tensor(float)":
            raise ValueError("ONNX input names/type do not match official export")
        shape = inputs[image_name].shape
        if len(shape) != 4 or shape[1:] != [3, 640, 640] or (isinstance(shape[0], int) and shape[0] != 1):
            raise ValueError("ONNX expects NCHW batch=1, 3x640x640")
        if kind in _RTDETR_IO and (inputs["orig_target_sizes"].type != "tensor(int64)" or
                len(inputs["orig_target_sizes"].shape) != 2 or inputs["orig_target_sizes"].shape[1] != 2):
            raise ValueError("orig_target_sizes must be int64 [1,2]")
        if {x.name for x in self.session.get_outputs()} != set(OUTPUT_NAMES[kind]):
            raise ValueError("ONNX output names do not match official export")
        if self.imported_candidate:
            expected_types = ({"labels": "tensor(int64)", "boxes": "tensor(float)", "scores": "tensor(float)"}
                              if kind in _RTDETR_IO else {"dets": "tensor(float)", "labels": "tensor(float)"})
            for value in self.session.get_outputs():
                wanted = self.manifest["output_shapes"][value.name]
                # Export metadata may retain symbolic dimensions. Concrete axes must
                # match now; every actual run is checked against the full contract.
                matches = (len(value.shape) == len(wanted) and
                           all(not isinstance(found, int) or found == expected
                               for found, expected in zip(value.shape, wanted)))
                if value.type != expected_types[value.name] or not matches:
                    raise ValueError("candidate runtime output metadata differs from manifest")
        if triton:
            from pathlib import Path
            from .model_artifact import sha256
            from .triton_detr import TRITON_MODELS, TritonSession
            # Identity: Triton must serve this sealed artifact, byte for byte (its repo is on the same host).
            # shortcut: checks the repo file, not what Triton loaded at start; restart Triton after changing the repo.
            served = Path(os.getenv("TRITON_REPO", "~/triton_repo_20261105")).expanduser() / TRITON_MODELS[kind] / "1" / "model.onnx"
            if not served.is_file() or sha256(served) != self.manifest["sha256"]:
                raise ValueError(f"Triton serves {served}, which is not the sealed artifact {self.artifact_path}")
            self.session, self.runtime = TritonSession(TRITON_MODELS[kind]), "triton"

    @property
    def settings(self):
        return {"kind": self.kind, "confidence": self.confidence, "top_k": self.top_k,
                "preprocess": "cv2-area-v2" if self.preprocess == "cv2" else "pil-bilinear-v1",
                "input_size": 640, "input_layout": "nchw", "label_map": "coco80_contiguous",
                "source_label_map": "coco91_sparse" if self.kind == "lwdetr_large" else "coco80_contiguous",
                "label_map_status": "assumed_unverified" if self.imported_candidate else "declared_by_export",
                "admission": "imported_candidate" if self.imported_candidate else "provenance_sealed",
                "artifact_contract": self.manifest}

    def prepare(self, bgr):
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
            raise ValueError("input must be nonempty uint8 HxWx3 BGR")
        height, width = bgr.shape[:2]
        if self.preprocess == "cv2":
            import cv2
            # Shrink first (multi-threaded C++, releases the GIL), then one fused
            # BGR->RGB, /255, HWC->NCHW on the small image. INTER_AREA averages
            # like PIL's antialiased bilinear, so the input changes very little.
            small = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA)
            tensor = cv2.dnn.blobFromImage(small, 1 / 255.0, swapRB=True)
            if self.kind == "lwdetr_large":
                tensor -= _IMAGENET_MEAN  # in place: the blob is a fresh array per frame
                tensor *= _IMAGENET_INV_STD
        else:
            from PIL import Image
            rgb = Image.fromarray(bgr[..., ::-1]).resize((640, 640), Image.Resampling.BILINEAR)
            tensor = np.asarray(rgb, dtype=np.float32) / 255.0
            if self.kind == "lwdetr_large":
                tensor = (tensor - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
            tensor = tensor.transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if self.kind in _RTDETR_IO:
            return {"images": tensor, "orig_target_sizes": np.array([[width, height]], dtype=np.int64)}
        return {"input": tensor}

    def infer(self, bgr):
        boxes, elapsed, _ = self.infer_detailed(bgr)
        return boxes, elapsed

    def infer_detailed(self, bgr):
        started = perf_counter()
        feed = self.prepare(bgr)
        prepared = perf_counter()
        # ORT owns each run's output buffers. Shared Cloud admission owns concurrency.
        output = self.session.run(list(OUTPUT_NAMES[self.kind]), feed)
        inferred = perf_counter()
        outputs = dict(zip(OUTPUT_NAMES[self.kind], output))
        for name, value in outputs.items():
            if list(value.shape) != self.manifest["output_shapes"][name]:
                raise ValueError(f"runtime output shape differs from artifact contract: {name}")
            wanted_dtype = np.int64 if self.kind in _RTDETR_IO and name == "labels" else np.float32
            if self.imported_candidate and value.dtype != wanted_dtype:
                raise ValueError(f"runtime output dtype differs from artifact contract: {name}")
        height, width = bgr.shape[:2]
        if self.kind in _RTDETR_IO:
            boxes = rtdetr_decode(outputs, width, height, self.confidence, self.top_k)
        else:
            boxes = lwdetr_decode(outputs, self.confidence, self.top_k)
        finished = perf_counter()
        return boxes, (finished - started) * 1000, {
            "cloud_preprocess_ms": (prepared - started) * 1000,
            "cloud_session_call_ms": (inferred - prepared) * 1000,
            "cloud_postprocess_ms": (finished - inferred) * 1000,
        }


class RtdetrV2SmallAdapter(CloudDetrAdapter):
    def __init__(self, onnx_path: str, **kwargs):
        super().__init__(onnx_path, kind="rtdetrv2_small", **kwargs)


class LwDetrLargeAdapter(CloudDetrAdapter):
    def __init__(self, onnx_path: str, **kwargs):
        super().__init__(onnx_path, kind="lwdetr_large", **kwargs)


class DfineXAdapter(CloudDetrAdapter):
    def __init__(self, onnx_path: str, **kwargs):
        super().__init__(onnx_path, kind="dfine_x", **kwargs)
