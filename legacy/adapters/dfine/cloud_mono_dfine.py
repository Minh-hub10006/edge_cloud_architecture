"""Cloud mono D-FINE: JPEG in → dfine_full.onnx (CUDA) → detection boxes."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def resolve_dfine_full_onnx() -> str:
    for key in ("DFINE_FULL_ONNX", "MONO_DFINE_ONNX"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
    for cand in (
        os.path.join(art, "onnx", "dfine_full.onnx"),
        os.path.expanduser("~/Parallelism_pipeline_AI_detection/onnx_export/dfine_full.onnx"),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "onnx_export",
            "dfine_full.onnx",
        ),
        "/home/stu0/Parallelism_pipeline_AI_detection/onnx_export/dfine_full.onnx",
        "/home/stu0/dfine_cloud_cuts/dfine_full.onnx",
    ):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "onnx_export", "dfine_full.onnx")
    )


def _letterbox_rgb(bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """BGR HWC → RGB NCHW float32 1x3xSxS in [0,1], simple resize (match common export)."""
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None, ...]


def _mono_detections(pred_boxes: np.ndarray, pred_logits: np.ndarray) -> tuple[list[dict], int, int, float]:
    """Canonical mono D-FINE sigmoid/top-k postprocess used by the old runtime."""
    boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    logits = np.asarray(pred_logits, dtype=np.float32)
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"D-FINE supports batch 1, got {logits.shape}")
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError(f"expected D-FINE logits [1,Q,C], got {pred_logits.shape}")
    queries = min(len(boxes), logits.shape[0])
    if queries == 0:
        return [], -1, -1, 0.0
    boxes, logits = boxes[:queries], logits[:queries]
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    flat_scores = scores.reshape(-1)
    top_k = max(1, int(os.getenv("DFINE_TOP_K", "20")))
    conf_thr = float(os.getenv("DFINE_CONF_THR", "0.10"))
    count = min(top_k, flat_scores.size)
    candidates = np.argpartition(flat_scores, -count)[-count:]
    candidates = candidates[np.argsort(-flat_scores[candidates])]
    class_count = scores.shape[1]
    best = int(candidates[0])
    best_query, best_class = divmod(best, class_count)
    best_score = float(flat_scores[best])
    box_format = os.getenv("DFINE_BOX_FORMAT", "cxcywh").strip().lower()
    if box_format not in {"cxcywh", "xyxy"}:
        raise ValueError("DFINE_BOX_FORMAT must be cxcywh or xyxy")
    detections = []
    for flat_index in candidates:
        confidence = float(flat_scores[int(flat_index)])
        if confidence < conf_thr:
            continue
        query, class_id = divmod(int(flat_index), class_count)
        a, b, c, d = [float(v) for v in boxes[query]]
        if box_format == "xyxy":
            left, top, right, bottom = a, b, c, d
        else:
            left, top, right, bottom = a - c / 2, b - d / 2, a + c / 2, b + d / 2
        detections.append({
            "class_id": class_id, "confidence": confidence,
            "left": max(0.0, min(1.0, left)), "top": max(0.0, min(1.0, top)),
            "right": max(0.0, min(1.0, right)), "bottom": max(0.0, min(1.0, bottom)),
        })
    return detections, best_query, best_class, best_score


class CloudMonoDFINE:
    """Runs full D-FINE ONNX on cloud GPU/CPU for mono JPEG uplink."""

    def __init__(self, onnx_path: Optional[str] = None):
        self.onnx_path = onnx_path or resolve_dfine_full_onnx()
        self.sess = None
        self.input_name = "images"
        self.runtime = "missing"
        if not HAS_ORT:
            print("[CloudMono] onnxruntime missing")
            return
        if not os.path.isfile(self.onnx_path):
            print(f"[CloudMono] ONNX not found: {self.onnx_path}")
            return
        from legacy.adapters.dfine.ort_providers import ensure_cuda_dlls

        ensure_cuda_dlls()
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
            self.input_name = self.sess.get_inputs()[0].name
            print(f"[CloudMono] {self.onnx_path} providers={used} runtime={self.runtime}")
            if self.runtime != "cuda":
                msg = (
                    "[CloudMono] WARNING: CUDA EP missing — mono ~16 FPS on CPU. "
                    "Start cloud with conda env `qnn` (onnxruntime-gpu)."
                )
                print(msg, flush=True)
                if require_cuda:
                    raise RuntimeError(
                        "CloudMonoDFINE requires CUDAExecutionProvider "
                        "(set REQUIRE_CUDA=0 to allow CPU fallback)"
                    )
        except Exception as e:
            print(f"[CloudMono] session failed: {e}")
            self.sess = None
            if require_cuda:
                raise

    def _run_model(self, inp):
        """Single inference hook shared by live and timed benchmark decoders."""
        return self.sess.run(None, {self.input_name: inp})

    def infer_bgr(self, bgr: np.ndarray) -> Tuple[float, str, Dict[str, Any]]:
        """Run the full Cloud model directly on a decoded RTSP BGR frame.

        This is the Phase 7 path: the video frame comes from CloudFrameCache;
        no JPEG encode, upload, or second image decode occurs.
        """
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("D-FINE input must be an HxWx3 BGR frame")
        t0 = time.perf_counter()
        if self.sess is None:
            raise RuntimeError("D-FINE ONNX session is unavailable; refusing mock inference")
        inp = _letterbox_rgb(bgr, 640)
        preprocess_ms = (time.perf_counter() - t0) * 1000
        stage_started = time.perf_counter()
        outs = self._run_model(inp)
        session_ms = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        logits, boxes = outs[0], outs[1]
        if logits.shape[-1] == 4 and boxes.shape[-1] != 4:
            logits, boxes = boxes, logits
        logits_f = np.asarray(logits, dtype=np.float32)
        det_boxes, best_q, _best_class, best_score = _mono_detections(boxes, logits_f)
        matched_id = f"mono_q{best_q}_s{best_score:.2f}"
        extras = {"boxes": det_boxes, "stages_ms": {
            "cloud_preprocess_ms": preprocess_ms,
            "cloud_session_call_ms": session_ms,
            "cloud_postprocess_ms": (time.perf_counter() - stage_started) * 1000,
        }}
        return time.perf_counter() - t0, matched_id, extras

    def decode_jpeg(self, jpeg_bytes: bytes) -> Tuple[float, str, Dict[str, Any]]:
        t0 = time.time()
        if self.sess is None:
            from real_data import require_real

            require_real("CloudMonoDFINE.decode_jpeg", "no dfine_full session")
            return time.time() - t0, "mock_mono", {"boxes": []}

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG decode failed")
        infer_s, matched_id, extras = self.infer_bgr(bgr)
        extras["pred_logits"] = None
        extras["pred_boxes"] = None
        return infer_s, matched_id, extras
