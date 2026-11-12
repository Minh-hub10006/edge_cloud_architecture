"""YOLO via ONNX Runtime — standard Ultralytics export [1,84,8400]."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from detectors.base import BaseLocalDetector, DetectionResult
from detectors.dfine.ort_providers import HAS_ORT, create_ort_session, runtime_label
from vis import draw_boxes

from inference.detection_decode import COCO_NAMES

_COCO = dict(enumerate(COCO_NAMES))   # all 80 names (13/11: other classes were stored as "cls28", "cls9", ...)


def resolve_yolo_ultralytics_onnx(name: str = "yolov8n.onnx") -> str:
    explicit = os.environ.get("YOLO_ONNX", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
    cand = os.path.join(art, "onnx", name)
    if os.path.isfile(cand):
        return cand
    return cand


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


class YOLOOrtDetector(BaseLocalDetector):
    """Ultralytics ONNX on ORT (CUDA on Windows GPU edges; QNN when available)."""

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        variant: str = "yolov8",
        imgsz: int = 640,
        score_thresh: float = 0.25,
        max_det: int = 20,
    ):
        self.variant = variant
        self.imgsz = int(os.environ.get("YOLO_IMGSZ", str(imgsz)))
        self.score_thresh = score_thresh
        self.max_det = max_det
        self.onnx_path = resolve_yolo_ultralytics_onnx(
            Path(onnx_path or "yolov8n.onnx").name if onnx_path else "yolov8n.onnx"
        )
        if onnx_path and os.path.isfile(onnx_path):
            self.onnx_path = onnx_path
        self.session = None
        self.runtime = "mock"
        self.providers: List[str] = []

        if HAS_ORT and os.path.isfile(self.onnx_path):
            qnn_bin = os.environ.get("YOLO_QNN_BIN", "").strip() or None
            self.session, self.runtime, self.providers = create_ort_session(
                self.onnx_path, label="YOLOOrt", qnn_context_bin=qnn_bin
            )
        else:
            print(f"[YOLOOrt] missing {self.onnx_path}")

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad[:nh, :nw] = resized
        tensor = pad.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return tensor, r, r

    def _postprocess(self, pred: np.ndarray, r: float) -> list[dict]:
        # pred: [1, 84, 8400] or [1, 8400, 84]
        if pred.ndim == 3 and pred.shape[1] == 84:
            pred = pred[0].T
        else:
            pred = pred[0]
        boxes_cxcywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_id = cls_scores.argmax(axis=1)
        conf = cls_scores[np.arange(len(cls_id)), cls_id]
        mask = conf >= self.score_thresh
        boxes_cxcywh = boxes_cxcywh[mask]
        conf = conf[mask]
        cls_id = cls_id[mask]
        if len(conf) == 0:
            return []
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
                "label": _COCO.get(cid, f"cls{cid}"),
            })
        return out

    def run(self, frame: np.ndarray, *, headless: bool = False) -> DetectionResult:
        t0 = time.time()
        tensor, rx, ry = self._preprocess(frame)
        r = rx

        if self.session is not None:
            in_name = self.session.get_inputs()[0].name
            raw = self.session.run(None, {in_name: tensor})[0]
            boxes_list = self._postprocess(raw, r)
        else:
            boxes_list = []

        num_det = len(boxes_list)
        if num_det:
            best = max(boxes_list, key=lambda b: b["confidence"])
            matched_id = f"{self.variant}_cls{best['class_id']}_{best['confidence']:.2f}"
        else:
            matched_id = f"{self.variant}_none"

        display_frame = frame if headless else draw_boxes(frame, boxes_list, color=(0, 255, 0))
        return DetectionResult(
            matched_id=matched_id,
            process_time_s=time.time() - t0,
            display_frame=display_frame,
            meta={
                "variant": self.variant,
                "path": self.onnx_path,
                "runtime": self.runtime,
                "providers": self.providers,
                "num_detections": num_det,
                "imgsz": self.imgsz,
                "backend": "ort",
            },
            boxes=boxes_list,
        )
