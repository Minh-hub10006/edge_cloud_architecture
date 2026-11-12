"""
Cloud Segment B + C — decoder on tier1 (P3, P4, P5).

Ported from Parallelism_pipeline_AI_detection/scripts/_server_pipeline_server.py
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from legacy.adapters.dfine.tier1_codec import unflatten_tier1, unpack_tier1_int8

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# Segment C inputs (names → shapes) — from _server_pipeline_server.py
C_HARDCODED = {
    "memory": (1, 8400, 256),
    "output": (1, 300, 256),
    "ref_points_detach": (1, 300, 4),
    "pred_corners_undetach": (1, 300, 132),
    "ref_points_initial": (1, 300, 4),
}

C_HARDCODED_SHAPES = C_HARDCODED  # alias for tier1_codec import compatibility


def _pred_boxes_to_detections(
    pred_boxes: np.ndarray, scores: np.ndarray, *, top_k: int = 50,
    conf_thr: float | None = None, class_ids: np.ndarray | None = None,
) -> list[dict]:
    """pred_boxes [1,Q,4] cxcywh normalized → detection dicts for gRPC feedback."""
    if conf_thr is None:
        conf_thr = float(os.environ.get("DFINE_CONF_THR", "0.12"))
    top_k = int(os.environ.get("DFINE_TOP_K", str(top_k)))
    pb = np.asarray(pred_boxes).reshape(-1, 4)
    sc = np.asarray(scores).reshape(-1)
    order = np.argsort(-sc)
    out: list[dict] = []
    for idx in order[:top_k]:
        conf = float(sc[idx])
        if conf < conf_thr:
            continue
        cx, cy, w, h = [float(x) for x in pb[idx]]
        out.append(
            {
                "class_id": 0 if class_ids is None else int(np.asarray(class_ids).reshape(-1)[idx]),
                "confidence": conf,
                "left": max(0.0, cx - w / 2),
                "top": max(0.0, cy - h / 2),
                "right": min(1.0, cx + w / 2),
                "bottom": min(1.0, cy + h / 2),
            }
        )
    return out


class CloudSegmentBC:
    """Runs segment_b.onnx + segment_c.onnx on cloud GPU/CPU."""

    def __init__(
        self,
        segment_b_path: Optional[str] = None,
        segment_c_path: Optional[str] = None,
        pipeline_root: Optional[str] = None,
    ):
        root = pipeline_root or os.environ.get("PIPELINE_AI_DETECTION_ROOT", "")
        self.segment_b_path = segment_b_path or os.path.join(
            root, "onnx_export", "segment_b.onnx"
        )
        self.segment_c_path = segment_c_path or os.path.join(
            root, "onnx_export", "segment_c.onnx"
        )
        self.sess_b = None
        self.sess_c = None
        self._pre_c_zeros = {
            nm: np.zeros(sh, dtype=np.float32) for nm, sh in C_HARDCODED.items()
        }

        if HAS_ORT and os.path.exists(self.segment_b_path) and os.path.exists(
            self.segment_c_path
        ):
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"[DFINE Cloud] B: {self.segment_b_path}")
            print(f"[DFINE Cloud] C: {self.segment_c_path}")
            self.sess_b = ort.InferenceSession(
                self.segment_b_path, sess_options=so, providers=providers
            )
            self.sess_c = ort.InferenceSession(
                self.segment_c_path, sess_options=so, providers=providers
            )
        else:
            from real_data import require_real

            require_real(
                "DFINECloudSegmentBC",
                "segment_b/c ONNX missing or onnxruntime absent — refuse mock decode",
            )
            print("[DFINE Cloud] segment_b/c ONNX missing or onnxruntime absent — mock decode")

    def _run_bc(self, P3: np.ndarray, P4: np.ndarray, P5: np.ndarray) -> List[np.ndarray]:
        outs = self.sess_b.run(None, {"P3": P3, "P4": P4, "P5": P5})
        out_by_shape = {tuple(arr.shape): arr for arr in outs}
        c_in = {}
        for nm, shape in C_HARDCODED.items():
            if shape in out_by_shape:
                c_in[nm] = out_by_shape[shape]
            else:
                c_in[nm] = self._pre_c_zeros[nm]
        return self.sess_c.run(None, c_in)

    def decode(self, tier1_flat: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
        """
        tier1_flat → (decode_time_s, matched_id, extras).
        extras may contain pred_logits, pred_boxes when ONNX available.
        """
        return self.decode_payload(np.asarray(tier1_flat, dtype=np.float32).tobytes(), layout="tier1")

    def decode_bytes(self, payload: bytes, *, layout: str = "tier1_int8") -> Tuple[float, str, Dict[str, Any]]:
        return self.decode_payload(payload, layout=layout)

    def decode_payload(self, payload: bytes, *, layout: str) -> Tuple[float, str, Dict[str, Any]]:
        t0 = time.time()
        if layout == "tier1_int8":
            tensors = unpack_tier1_int8(payload)
        else:
            flat = np.frombuffer(payload, dtype=np.float32)
            tensors = unflatten_tier1(flat)

        if self.sess_b is not None and self.sess_c is not None:
            outs = self._run_bc(tensors["P3"], tensors["P4"], tensors["P5"])
            logits, boxes = outs[-2], outs[-1]
            # Per-class sigmoid then take best class score (not sigmoid(max logit))
            logits_f = np.asarray(logits, dtype=np.float32)
            probs = 1.0 / (1.0 + np.exp(-logits_f))
            scores = probs.max(axis=-1).squeeze()
            best_q = int(np.asarray(scores).reshape(-1).argmax())
            best_score = float(np.asarray(scores).reshape(-1)[best_q])
            matched_id = f"dfine_q{best_q}_s{best_score:.2f}"
            det_boxes = _pred_boxes_to_detections(boxes, scores)
            extras = {
                "pred_logits": logits,
                "pred_boxes": boxes,
                "boxes": det_boxes,
            }
        else:
            from real_data import require_real

            require_real(
                "DFINECloudSegmentBC.decode",
                "ORT sessions missing — refuse mock_dfine_cloud matched_id",
            )
            time.sleep(0.012)
            matched_id = "mock_dfine_cloud"
            extras = {}

        return time.time() - t0, matched_id, extras
