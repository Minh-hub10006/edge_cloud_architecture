"""YOLOv8 on QCS6490 NPU via ORT QNN (precompiled_qnn_onnx from AI Hub)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from detectors.base import BaseLocalDetector, DetectionResult
from detectors.dfine.ort_providers import HAS_ORT, create_ort_session
from vis import draw_boxes

_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
}


def _pipeline_root() -> str:
    return os.environ.get(
        "EDGE_ARTIFACTS",
        os.path.expanduser("~/edge_artifacts"),
    )


def resolve_yolo_onnx(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for key in ("YOLO_ONNX",):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    art = _pipeline_root()
    for name in ("yolov8_det.onnx", "yolov8n_int8.onnx"):
        cand = os.path.join(art, "onnx", name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(art, "onnx", "yolov8_det.onnx")


def resolve_yolo_qnn_bin(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for key in ("YOLO_QNN_BIN",):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    art = _pipeline_root()
    for name in ("yolov8_det_qairt_context.bin", "yolov8n_qnn_int8.bin"):
        cand = os.path.join(art, "npu", name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(art, "npu", "yolov8_det_qairt_context.bin")


def _dequant(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    return (arr.astype(np.float32) - float(zero_point)) * float(scale)


def _load_labels(onnx_dir: Path) -> dict[int, str]:
    labels_path = onnx_dir / "labels.txt"
    if not labels_path.is_file():
        return dict(_COCO_NAMES)
    names: dict[int, str] = {}
    for i, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines()):
        label = line.strip()
        if label:
            names[i] = label
    return names or dict(_COCO_NAMES)


class YOLONpuDetector(BaseLocalDetector):
    """YOLOv8n INT8 on HTP — AI Hub precompiled_qnn_onnx export."""

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        qnn_bin: Optional[str] = None,
        imgsz: int = 640,
        score_thresh: float = 0.25,
        max_det: int = 20,
        variant: str = "yolov8",
    ):
        self.variant = variant
        self.imgsz = int(os.environ.get("YOLO_IMGSZ", str(imgsz)))
        self.score_thresh = score_thresh
        self.max_det = max_det
        self.onnx_path = resolve_yolo_onnx(onnx_path)
        self.qnn_bin = resolve_yolo_qnn_bin(qnn_bin)
        self.session = None
        self.runtime = "mock"
        self.providers: List[str] = []
        self.names = _load_labels(Path(self.onnx_path).parent)

        if HAS_ORT and os.path.isfile(self.onnx_path):
            self.session, self.runtime, self.providers = create_ort_session(
                self.onnx_path,
                label="YOLONpu",
                qnn_context_bin=self.qnn_bin if os.path.isfile(self.qnn_bin) else None,
            )
        else:
            print(
                f"[YOLONpu] onnx missing ({self.onnx_path}) or ORT unavailable"
            )

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = resized[np.newaxis, ...].astype(np.uint8)
        sx = w / self.imgsz
        sy = h / self.imgsz
        return tensor, sx, sy

    def _postprocess(
        self,
        outs: dict[str, np.ndarray],
        sx: float,
        sy: float,
    ) -> list[dict]:
        boxes = outs.get("boxes")
        scores = outs.get("scores")
        class_idx = outs.get("class_idx")
        if boxes is None or scores is None:
            return []

        boxes_f = _dequant(boxes[0], scale=3.1009655, zero_point=25)
        scores_f = _dequant(scores[0], scale=0.00390625, zero_point=0)
        if class_idx is not None:
            cls = class_idx[0].astype(np.int32)
        else:
            cls = np.argmax(scores_f, axis=-1) if scores_f.ndim > 1 else np.zeros(len(scores_f), dtype=np.int32)

        if scores_f.ndim > 1:
            conf = scores_f.max(axis=-1)
            if class_idx is None:
                cls = scores_f.argmax(axis=-1)
        else:
            conf = scores_f

        keep = np.where(conf >= self.score_thresh)[0]
        if keep.size == 0:
            return []
        order = keep[np.argsort(-conf[keep])][: self.max_det]

        result: list[dict] = []
        for i in order:
            x1, y1, x2, y2 = boxes_f[i].tolist()
            cid = int(cls[i]) if np.ndim(cls) else int(cls)
            c = float(conf[i])
            result.append({
                "class_id": cid,
                "confidence": c,
                "left": x1 * sx,
                "top": y1 * sy,
                "right": x2 * sx,
                "bottom": y2 * sy,
                "label": self.names.get(cid, f"cls{cid}"),
            })
        return result

    def run(self, frame: np.ndarray, *, headless: bool = False) -> DetectionResult:
        t0 = time.time()
        tensor, sx, sy = self._preprocess(frame)

        if self.session is not None:
            in_name = self.session.get_inputs()[0].name
            raw = self.session.run(None, {in_name: tensor})
            out_map = {o.name: v for o, v in zip(self.session.get_outputs(), raw)}
            boxes_list = self._postprocess(out_map, sx, sy)
        else:
            boxes_list = []

        num_det = len(boxes_list)
        if num_det > 0:
            best = max(boxes_list, key=lambda b: b["confidence"])
            matched_id = f"{self.variant}_cls{best['class_id']}_{best['confidence']:.2f}"
        else:
            matched_id = f"{self.variant}_none"

        display_frame = frame if headless else draw_boxes(frame, boxes_list, color=(0, 255, 0))
        meta = {
            "variant": self.variant,
            "path": self.onnx_path,
            "runtime": self.runtime,
            "providers": self.providers,
            "num_detections": num_det,
            "imgsz": self.imgsz,
            "backend": "npu",
        }
        return DetectionResult(
            matched_id=matched_id,
            process_time_s=time.time() - t0,
            display_frame=display_frame,
            meta=meta,
            boxes=boxes_list,
        )
