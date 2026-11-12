"""Cloud stem cut decoder: INT8 stem tensor → cloud_stem.onnx (CUDA) → boxes."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from detectors.dfine.cloud_segment_bc import _pred_boxes_to_detections
from detectors.dfine.stem_codec import STEM_SHAPE, unpack_stem

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def resolve_cloud_stem_onnx() -> str:
    for key in ("CLOUD_STEM_ONNX", "STEM_CLOUD_ONNX"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    for cand in (
        os.path.expanduser("~/dfine_cloud_cuts/cloud_stem.onnx"),
        os.path.expanduser("~/cuts_e2e/cloud_stem.onnx"),
        "/home/stu0/dfine_cloud_cuts/cloud_stem.onnx",
    ):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return os.path.expanduser("~/dfine_cloud_cuts/cloud_stem.onnx")


class CloudStemCut:
    """Runs cloud_stem.onnx on CUDA for stem-cut uplink."""

    def __init__(self, onnx_path: Optional[str] = None):
        self.onnx_path = onnx_path or resolve_cloud_stem_onnx()
        self.sess = None
        self.input_name = ""
        self.runtime = "missing"
        if not HAS_ORT:
            print("[CloudStem] onnxruntime missing")
            return
        if not os.path.isfile(self.onnx_path):
            print(f"[CloudStem] ONNX not found: {self.onnx_path}")
            return
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        require_cuda = os.environ.get("REQUIRE_CUDA", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        try:
            self.sess = ort.InferenceSession(
                self.onnx_path, sess_options=so, providers=providers
            )
            used = self.sess.get_providers()
            self.runtime = "cuda" if used and "CUDA" in used[0] else "cpu"
            input_meta = self.sess.get_inputs()[0]
            self.input_name = input_meta.name
            input_shape = tuple(input_meta.shape)
            expected_shape = tuple(STEM_SHAPE)
            if len(input_shape) == len(expected_shape) and any(
                actual != expected for actual, expected in zip(input_shape, expected_shape)
                if isinstance(actual, int)
            ):
                raise RuntimeError(
                    f"cloud_stem input shape {input_shape} is incompatible with "
                    f"edge stem contract {expected_shape}"
                )
            outputs = self.sess.get_outputs()
            if len(outputs) < 2:
                raise RuntimeError(
                    f"cloud_stem must return logits and boxes, got {[o.name for o in outputs]}"
                )
            print(
                f"[CloudStem] {self.onnx_path} input={input_meta.name}:{input_shape} "
                f"outputs={[o.name for o in outputs]} providers={used} runtime={self.runtime}"
            )
            if self.runtime != "cuda" and require_cuda:
                raise RuntimeError(
                    "CloudStemCut requires CUDAExecutionProvider "
                    "(set REQUIRE_CUDA=0 to allow CPU fallback)"
                )
        except Exception as e:
            print(f"[CloudStem] session failed: {e}")
            self.sess = None
            if require_cuda:
                raise

    def _run_model(self, stem: np.ndarray):
        """Inference hook used by the benchmark timing subclass."""
        return self.sess.run(None, {self.input_name: stem})

    def decode_bytes(
        self, payload: bytes, *, layout: str = "cut_stem_int8"
    ) -> Tuple[float, str, Dict[str, Any]]:
        t0 = time.time()
        if self.sess is None:
            from real_data import require_real

            require_real("CloudStemCut.decode_bytes", "no cloud_stem session")
            return time.time() - t0, "mock_stem", {"boxes": []}

        stem = unpack_stem(payload, layout=layout)
        outs = self._run_model(stem.astype(np.float32))
        logits, boxes = outs[0], outs[1]
        if logits.shape[-1] == 4 and boxes.shape[-1] != 4:
            logits, boxes = boxes, logits
        logits_f = np.asarray(logits, dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits_f))
        scores = probs.max(axis=-1).squeeze()
        best_q = int(np.asarray(scores).reshape(-1).argmax())
        best_score = float(np.asarray(scores).reshape(-1)[best_q])
        matched_id = f"stem_q{best_q}_s{best_score:.2f}"
        det_boxes, _best_q, _best_class, _best_score = _pred_boxes_to_detections(
            boxes, scores
        )
        extras = {"boxes": det_boxes, "pred_logits": logits, "pred_boxes": boxes}
        return time.time() - t0, matched_id, extras
