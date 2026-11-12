"""YOLOv8 — fully local detection on edge."""

import os

from detectors.yolo.base_yolo import YOLODetector

DEFAULT_WEIGHTS = "models/yolov8n.pt"


class YOLOv8Detector(YOLODetector):
    def __init__(self, model_path: str = DEFAULT_WEIGHTS, **kwargs):
        kwargs.pop("model_path", None)
        backend = os.environ.get("YOLO_BACKEND", "").strip().lower()
        if backend in ("snpe", "snpe_dlc", "dlc"):
            from detectors.yolo.yolo_snpe import YOLOSnpeDetector

            self._delegate = YOLOSnpeDetector(variant="yolov8", **kwargs)
            self._use_npu = True
            self.imgsz = self._delegate.imgsz
            return
        if backend in ("npu", "qnn", "ort_qnn"):
            from detectors.yolo.yolo_npu import YOLONpuDetector

            self._delegate = YOLONpuDetector(variant="yolov8", **kwargs)
            self._use_npu = True
            self.imgsz = self._delegate.imgsz
            return
        if backend in ("ort", "onnx", "onnxruntime"):
            from detectors.yolo.yolo_ort import YOLOOrtDetector

            self._delegate = YOLOOrtDetector(variant="yolov8", **kwargs)
            self._use_npu = False
            self.imgsz = self._delegate.imgsz
            return
        raise RuntimeError(
            f"YOLO_BACKEND={backend!r} unsupported. "
            "Set YOLO_BACKEND=snpe_dlc|npu (edge-246) or ort (Windows GPU edge)."
        )

    def run(self, frame, *, headless: bool = False):
        return self._delegate.run(frame, headless=headless)
