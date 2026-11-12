"""Shared Ultralytics YOLO runner for v8 / 11 / 26 variants."""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from detectors.base import BaseLocalDetector, DetectionResult
from detectors.demo_boxes import MOCK_TRAFFIC_BOXES
from real_data import require_real
from vis import draw_boxes

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None  # type: ignore


def _resolve_weights(model_path: str) -> str:
    if os.path.isabs(model_path) and os.path.isfile(model_path):
        return model_path
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    for base in (os.getcwd(), root):
        candidate = os.path.join(base, model_path)
        if os.path.isfile(candidate):
            return candidate
    return model_path


class YOLODetector(BaseLocalDetector):
    def __init__(
        self,
        model_path: str,
        variant: str = "yolov8",
        mock_sleep_s: float = 0.02,
        imgsz: int | None = None,
    ):
        self.model_path = _resolve_weights(model_path)
        self.variant = variant
        self.mock_sleep_s = mock_sleep_s
        self.imgsz = imgsz if imgsz is not None else int(os.environ.get("YOLO_IMGSZ", "640"))
        self.model = None

        if HAS_ULTRALYTICS and os.path.isfile(self.model_path):
            try:
                print(f"[{variant}] Loading weights: {self.model_path}")
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"[{variant}] Load failed ({e}), using demo overlay")
                self.model = None
        else:
            print(
                f"[{variant}] Demo overlay (weights not found: {self.model_path})"
            )

    def run(self, frame: np.ndarray, *, headless: bool = False) -> DetectionResult:
        t0 = time.time()

        if self.model is not None:
            results = self.model(
                frame,
                verbose=False,
                imgsz=self.imgsz,
                max_det=20,
            )
            display_frame = frame if headless else results[0].plot()
            ultra_boxes = results[0].boxes
            num_det = 0 if ultra_boxes is None else len(ultra_boxes)
            boxes_list = []
            if num_det > 0:
                cls_id = int(ultra_boxes.cls[0].item())
                conf = float(ultra_boxes.conf[0].item())
                matched_id = f"{self.variant}_cls{cls_id}_{conf:.2f}"
                names = results[0].names
                for i in range(num_det):
                    x1, y1, x2, y2 = ultra_boxes.xyxy[i].tolist()
                    cid = int(ultra_boxes.cls[i].item())
                    boxes_list.append({
                        "class_id": cid,
                        "confidence": float(ultra_boxes.conf[i].item()),
                        "left": x1, "top": y1, "right": x2, "bottom": y2,
                        "label": names.get(cid, f"cls{cid}"),
                    })
            else:
                matched_id = f"{self.variant}_none"
            if not headless and num_det > 0 and num_det <= 5:
                print(
                    f"[{self.variant}] frame detections={num_det} top={matched_id}"
                )
            meta = {
                "variant": self.variant,
                "path": self.model_path,
                "num_detections": num_det,
                "imgsz": self.imgsz,
            }
        else:
            require_real(
                "YOLO",
                f"weights missing or ultralytics unavailable ({self.model_path})",
            )
            time.sleep(self.mock_sleep_s)
            display_frame = draw_boxes(
                frame, MOCK_TRAFFIC_BOXES, color=(0, 255, 0)
            )
            best = max(MOCK_TRAFFIC_BOXES, key=lambda b: b["confidence"])
            matched_id = (
                f"{self.variant}_cls{best['class_id']}_{best['confidence']:.2f}"
            )
            meta = {
                "variant": self.variant,
                "path": self.model_path,
                "num_detections": len(MOCK_TRAFFIC_BOXES),
                "demo": True,
            }
            boxes_list = list(MOCK_TRAFFIC_BOXES)

        return DetectionResult(
            matched_id=matched_id,
            process_time_s=time.time() - t0,
            display_frame=display_frame,
            meta=meta,
            boxes=boxes_list,
        )
