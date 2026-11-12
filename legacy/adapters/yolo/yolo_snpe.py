"""YOLOv8 on SNPE DLC (QAIRT 2.43 INT8) — warm storage cache, burst profile."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from legacy.adapters.npu.snpe_dlc_session import SnpeDlcSession, resolve_dlc


class BaseLocalDetector:
    """Minimal local base retained for this isolated adapter snapshot."""


@dataclass(slots=True)
class DetectionResult:
    matched_id: str
    process_time_s: float
    display_frame: np.ndarray
    meta: dict
    boxes: list[dict]


def draw_boxes(frame: np.ndarray, boxes: list[dict], color=(0, 255, 0)) -> np.ndarray:
    output = frame.copy()
    for box in boxes:
        x1, y1 = int(box["left"]), int(box["top"])
        x2, y2 = int(box["right"]), int(box["bottom"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    return output

_COCO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45, top_k: int = 20) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < top_k:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


class YOLOSnpeDetector(BaseLocalDetector):
    def __init__(
        self,
        dlc_path: Optional[str] = None,
        variant: str = "yolov8",
        imgsz: int = 640,
        score_thresh: float = 0.25,
        max_det: int = 20,
    ):
        self.variant = variant
        self.imgsz = int(os.environ.get("YOLO_IMGSZ", str(imgsz)) if variant != "plate" else os.environ.get("PLATE_IMGSZ", str(imgsz)))
        self.score_thresh = float(os.environ.get("YOLO_SCORE_THRESH", str(score_thresh)))
        self.max_det = int(os.environ.get("YOLO_MAX_DET", str(max_det)))
        if variant == "plate":
            name = "plate_yolov8n_320"
            explicit = dlc_path or os.environ.get("PLATE_DLC")
        else:
            name = "yolov8n_320" if self.imgsz <= 320 else "yolov8n_640"
            # YOLO_DLC may point at 640 while YOLO_IMGSZ=320 — only honor if basename matches.
            env_dlc = (os.environ.get("YOLO_DLC") or "").strip()
            if dlc_path:
                explicit = dlc_path
            elif env_dlc and name in os.path.basename(env_dlc):
                explicit = env_dlc
            else:
                explicit = None
        self.dlc_path = resolve_dlc(name, explicit)
        # Input layout is an artifact contract, never a generic "split" heuristic.
        # The verified traffic100 split artifact is NCHW (the benchmark wrapper
        # explicitly used NCHW); treating it as NHWC produces valid NPU calls but
        # zero detections in WebLive.
        # Plate: the ONNX (scripts/export_plate_onnx.py, ORT on a GPU Edge) is NCHW, but the DLC converted from it takes
        # NHWC (the SNPE converter's default). 11/11 on 246, 52 VN vehicle crops vs the float model: fed NCHW the DLC gave
        # garbage (box IoU 0/49, 11 boxes on plate-less crops - every plate crop since 8/11, v50 too); fed NHWC 49/49
        # at IoU >= 0.5 (median 0.93), none on plate-less crops. YOLO_INPUT_LAYOUT belongs to the vehicle YOLO.
        if variant == "plate":
            layout = "nchw" if self.dlc_path.lower().endswith(".onnx") else "nhwc"
        else:
            layout = (os.environ.get("YOLO_INPUT_LAYOUT") or "").strip().lower()
        if not layout:
            layout = self._default_input_layout(self.dlc_path)
        self.input_layout = layout if layout in ("nchw", "nhwc") else "nchw"
        self.session: Optional[SnpeDlcSession] = None
        self.runtime = "snpe_dsp_dlc"
        self.infer_mode = "unavailable"
        self._pad: Optional[np.ndarray] = None
        self._tensor: Optional[np.ndarray] = None
        if self.dlc_path.lower().endswith(".onnx"):  # GPU Edge: same graph source, ORT CUDA
            from inference.ort_session import OrtCudaSession
            self.session = OrtCudaSession(self.dlc_path)
            self.infer_mode = self.session.mode
            self.runtime = "ort_cuda"
            return
        try:
            # YOLOV8N_SNPE_GRAPH (host config): strict DSP through the cache-off v3 daemon, as YOLO11s. Edge252's
            # legacy daemon path (init cache) silently ran YOLOv8n on the CPU (~300% CPU, Oct 25-28).
            # The plate detector runs beside YOLOv8n in one process: it must not inherit YOLOv8n's graph.
            graph = (os.environ.get("PLATE_SNPE_GRAPH" if variant == "plate" else "YOLOV8N_SNPE_GRAPH") or "").strip()
            self.session = (SnpeDlcSession(self.dlc_path, runtime="dsp", require_daemon=True,
                                           output_tensors=("boxes", "scores"), input_bytes=320 * 320 * 3 * 4,
                                           graph_name=graph)
                            if graph else SnpeDlcSession(self.dlc_path))
            self.infer_mode = self.session.mode
            self.runtime = f"snpe_dsp_dlc_{self.infer_mode}"
        except FileNotFoundError as e:
            print(f"[YOLOSnpe] {e}")

    @staticmethod
    def _default_input_layout(dlc_path: Optional[str]) -> str:
        """Return the verified default layout for a known DLC artifact."""
        base = os.path.basename(dlc_path or "").lower()
        if "split_traffic100" in base:
            return "nchw"
        return "nhwc" if "onnx" in base else "nchw"
    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame_bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        if self._pad is None:
            self._pad = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
            if self.input_layout == "nhwc":
                self._tensor = np.empty((1, self.imgsz, self.imgsz, 3), dtype=np.float32)
            else:
                self._tensor = np.empty((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        self._pad.fill(114)
        # Channel permutation commutes with resize. Convert only the small
        # model input, avoiding a full-resolution RGB allocation per frame.
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._pad[:nh, :nw] = resized
        if self.input_layout == "nhwc":
            np.multiply(self._pad, 1.0 / 255.0, out=self._tensor[0], casting="unsafe")
        else:
            chw = self._pad.transpose(2, 0, 1)
            np.multiply(chw, 1.0 / 255.0, out=self._tensor[0], casting="unsafe")
        return self._tensor, r

    @staticmethod
    def _pick_outputs(outs: dict) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (pred_or_boxes, scores_or_None). Supports split boxes+scores DLC."""
        if not outs:
            return np.zeros((0,), dtype=np.float32), None
        # Prefer explicit split names from scripts/split_yolo_onnx_outputs.py
        lower = {str(k).lower(): v for k, v in outs.items()}
        boxes_k = next((k for k in lower if "box" in k), None)
        scores_k = next((k for k in lower if "score" in k or "cls" in k), None)
        if boxes_k and scores_k:
            return np.asarray(lower[boxes_k], dtype=np.float32), np.asarray(lower[scores_k], dtype=np.float32)
        if len(outs) >= 2:
            # Heuristic: smaller first-dim channel count is boxes (4) vs scores (80)
            items = [(k, np.asarray(v, dtype=np.float32).reshape(-1)) for k, v in outs.items()]
            items.sort(key=lambda kv: kv[1].size)
            # 4*N then 80*N
            return items[0][1], items[-1][1]
        return np.asarray(next(iter(outs.values())), dtype=np.float32), None

    def _postprocess(
        self,
        pred: np.ndarray,
        r: float,
        scores: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Decode YOLO SNPE output.

        Prefer split DLC (boxes 4xN + scores 80xN) so INT8 keeps independent scales.
        Legacy single-tensor DLC: channel-first (84, N). Do NOT invent geometric boxes
        when the INT8 class head is zero — that floods the frame with junk.
        """
        n = 8400 if self.imgsz >= 640 else 2100

        if scores is not None:
            b = np.asarray(pred, dtype=np.float32).reshape(-1)
            s = np.asarray(scores, dtype=np.float32).reshape(-1)
            # boxes: (4,N) or (N,4); scores: (C,N) or (N,C)
            if b.size == 4 * n:
                boxes_cxcywh = b.reshape(4, n).T
            elif b.size % 4 == 0:
                n = b.size // 4
                boxes_cxcywh = b.reshape(4, n).T
            else:
                boxes_cxcywh = b.reshape(-1, 4)
                n = boxes_cxcywh.shape[0]
            if s.size % n == 0:
                c = max(s.size // n, 1)
                cls_scores = s.reshape(c, n).T
            else:
                cls_scores = s.reshape(n, -1)
            if cls_scores.ndim == 1:
                cls_scores = cls_scores.reshape(-1, 1)
        else:
            arr = np.asarray(pred, dtype=np.float32).reshape(-1)
            nc = 84
            for cand in (84, 5, 6, 4):
                if arr.size == cand * n:
                    nc = cand
                    break
            else:
                n = max(arr.size // 84, 1)
                nc = max(arr.size // n, 1)
            if arr.size == nc * n:
                pred2d = arr.reshape(nc, n).T
            else:
                pred2d = arr.reshape(-1, nc)
            boxes_cxcywh = pred2d[:, :4]
            cls_scores = (
                pred2d[:, 4:]
                if pred2d.shape[1] > 4
                else np.ones((pred2d.shape[0], 1), dtype=np.float32)
            )

        if float(cls_scores.max()) < 1e-5:
            return []

        cls_id = cls_scores.argmax(axis=1)
        conf = cls_scores[np.arange(len(cls_id)), cls_id]
        mask = conf >= self.score_thresh
        if not mask.any():
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        conf = conf[mask]
        cls_id = cls_id[mask]
        cx, cy, bw, bh = boxes_cxcywh.T
        x1 = (cx - bw / 2) / r
        y1 = (cy - bh / 2) / r
        x2 = (cx + bw / 2) / r
        y2 = (cy + bh / 2) / r
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes_xyxy, conf, top_k=self.max_det)
        out: list[dict] = []
        for i in keep:
            cid = int(cls_id[i])
            out.append({
                "class_id": cid,
                "confidence": float(conf[i]),
                "left": float(x1[i]),
                "top": float(y1[i]),
                "right": float(x2[i]),
                "bottom": float(y2[i]),
                "label": ("plate" if self.variant == "plate" else _COCO.get(cid, f"cls{cid}")),
            })
        return out

    def decode_outputs(self, outs: dict, r: float) -> list[dict]:
        """Canonical SNPE output decoder shared by WebLive and benchmark."""
        if not outs:
            return []
        boxes_t, scores_t = self._pick_outputs(outs)
        return self._postprocess(boxes_t, r, scores_t)

    def detect_boxes(self, frame_bgr: np.ndarray) -> list[dict]:
        """Headless inference — returns box dicts with left/top/right/bottom/confidence."""
        tensor, r = self._preprocess(frame_bgr)
        if self.session is None:
            return []
        _, outs = self.session.infer_raw(tensor)
        if not outs:
            return []
        return self.decode_outputs(outs, r)

    def run(self, frame: np.ndarray, *, headless: bool = False) -> DetectionResult:
        t0 = time.perf_counter()
        tensor, r = self._preprocess(frame)
        boxes_list: list[dict] = []
        if self.session is not None:
            _, outs = self.session.infer_raw(tensor)
            if outs:
                boxes_list = self.decode_outputs(outs, r)
        num_det = len(boxes_list)
        if headless:
            matched_id = f"{self.variant}_{num_det}"
        elif num_det:
            best = max(boxes_list, key=lambda b: b["confidence"])
            matched_id = f"{self.variant}_cls{best['class_id']}_{best['confidence']:.2f}"
        else:
            matched_id = f"{self.variant}_none"
        display_frame = frame if headless else draw_boxes(frame, boxes_list, color=(0, 255, 0))
        return DetectionResult(
            matched_id=matched_id,
            process_time_s=time.perf_counter() - t0,
            display_frame=display_frame,
            meta={
                "variant": self.variant,
                "path": self.dlc_path,
                "runtime": self.runtime,
                "num_detections": num_det,
                "imgsz": self.imgsz,
                "backend": "snpe_dlc",
                "infer_mode": self.infer_mode,
            },
            boxes=boxes_list,
        )
