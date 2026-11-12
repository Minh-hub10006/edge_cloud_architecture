"""Thin adapters around the copied, model-specific production runners.

The platform owns identity, transport and normalized results. These adapters
own only model loading, preprocessing and detector-specific box conversion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from contracts import Box


class FrameDetector(Protocol):
    model_id: str

    def infer(self, bgr: np.ndarray) -> tuple[list[Box], float]:
        """Return source-normalized boxes and measured inference milliseconds."""


@dataclass(frozen=True)
class PreparedYoloFrame:
    tensor: np.ndarray
    ratio: float
    width: int
    height: int
    stages_ms: dict[str, float]


def _normalize_xyxy(item: dict[str, Any], width: int, height: int) -> Box | None:
    x1 = min(max(float(item["left"]), 0.0), float(width)) / width
    y1 = min(max(float(item["top"]), 0.0), float(height)) / height
    x2 = min(max(float(item["right"]), 0.0), float(width)) / width
    y2 = min(max(float(item["bottom"]), 0.0), float(height)) / height
    if x2 <= x1 or y2 <= y1:
        return None
    class_id = int(item.get("class_id", 0))
    class_name = str(item.get("label") or item.get("class_name") or f"class_{class_id}")
    return Box(
        left=x1, top=y1, right=x2, bottom=y2,
        confidence=min(max(float(item["confidence"]), 0.0), 1.0),
        class_id=class_id, class_name=class_name,
    )


class YoloSnpeAdapter:
    """YOLOv8 INT8 DLC adapter for Edge 246/Qualcomm SNPE."""

    def __init__(
        self, *, dlc_path: str | None = None, imgsz: int = 320,
        score_threshold: float = 0.25, input_layout: str = "nchw",
    ) -> None:
        if input_layout not in {"nchw", "nhwc"}:
            raise ValueError("YOLO input_layout must be nchw or nhwc")
        artifact_root = Path(os.environ.get("EDGE_ARTIFACTS", str(Path.home() / "edge_artifacts")))
        selected_dlc = Path(dlc_path).expanduser() if dlc_path else (
            artifact_root / "npu243" / "yolov8n_320_int8_split_traffic100.dlc"
        )
        if not selected_dlc.is_file():
            raise FileNotFoundError(f"verified YOLO DLC not found: {selected_dlc}")
        # The old Edge runner reads these from the process environment even
        # when constructor arguments are provided. CLI parameters are the
        # authoritative artifact contract for this isolated process.
        os.environ["YOLO_IMGSZ"] = str(imgsz)
        os.environ["YOLO_INPUT_LAYOUT"] = input_layout
        os.environ["YOLO_SCORE_THRESH"] = str(score_threshold)
        from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector

        self.detector = YOLOSnpeDetector(
            dlc_path=str(selected_dlc), variant="yolov8", imgsz=imgsz,
            score_thresh=score_threshold,
        )
        if self.detector.session is None:
            raise RuntimeError(
                f"YOLO SNPE DLC/session unavailable: {self.detector.dlc_path}; "
                "refusing mock/empty inference"
            )
        self.model_id = Path(self.detector.dlc_path).stem

    def infer(self, bgr: np.ndarray) -> tuple[list[Box], float]:
        boxes, elapsed_ms, _ = self.infer_prepared(self.prepare_frame(bgr))
        return boxes, elapsed_ms

    def prepare_frame(self, bgr: np.ndarray) -> PreparedYoloFrame:
        """One preparation lane; its owned tensor can overlap SNPE inference."""
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("YOLO input must be an HxWx3 BGR frame")
        started = time.perf_counter()
        tensor, ratio = self.detector._preprocess(bgr)
        prepared_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        tensor = tensor.copy()  # Never hand mutable detector scratch to another lane.
        copy_ms = (time.perf_counter() - started) * 1000
        height, width = bgr.shape[:2]
        return PreparedYoloFrame(tensor, ratio, width, height,
            {"edge_preprocess_ms": prepared_ms, "edge_tensor_copy_ms": copy_ms})

    def infer_prepared(self, prepared: PreparedYoloFrame):
        """Only one caller may run SNPE; preparation may run concurrently."""
        stages = dict(prepared.stages_ms)
        started = time.perf_counter()
        reported_s, outputs = self.detector.session.infer_raw(prepared.tensor)
        stages["edge_session_call_ms"] = (time.perf_counter() - started) * 1000
        stages["edge_npu_reported_ms"] = float(reported_s) * 1000
        started = time.perf_counter()
        raw = self.detector.decode_outputs(outputs, prepared.ratio)
        stages["edge_postprocess_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        boxes = [box for item in raw if (box := _normalize_xyxy(item, prepared.width, prepared.height))]
        stages["edge_normalize_ms"] = (time.perf_counter() - started) * 1000
        elapsed_ms = sum(value for key, value in stages.items() if key != "edge_npu_reported_ms")
        return boxes, elapsed_ms, stages


class DfineCloudAdapter:
    """Full D-FINE ONNX adapter for a Cloud-decoded RTSP frame."""

    def __init__(self, onnx_path: str | None = None) -> None:
        from legacy.adapters.dfine.cloud_mono_dfine import CloudMonoDFINE

        self.detector = CloudMonoDFINE(onnx_path=onnx_path)
        if self.detector.sess is None:
            raise RuntimeError(
                f"D-FINE ONNX session unavailable: {self.detector.onnx_path}; "
                "refusing mock inference"
            )
        self.model_id = Path(self.detector.onnx_path).stem
        self.runtime = self.detector.runtime

    def infer(self, bgr: np.ndarray) -> tuple[list[Box], float]:
        boxes, elapsed_ms, _ = self.infer_detailed(bgr)
        return boxes, elapsed_ms

    def infer_detailed(self, bgr: np.ndarray):
        started = time.perf_counter()
        model_seconds, _matched_id, output = self.detector.infer_bgr(bgr)
        elapsed_ms = max(model_seconds * 1000.0, (time.perf_counter() - started) * 1000.0)
        boxes = []
        normalize_started = time.perf_counter()
        for item in output.get("boxes", []):
            left = min(max(float(item["left"]), 0.0), 1.0)
            top = min(max(float(item["top"]), 0.0), 1.0)
            right = min(max(float(item["right"]), 0.0), 1.0)
            bottom = min(max(float(item["bottom"]), 0.0), 1.0)
            if right <= left or bottom <= top:
                continue
            boxes.append(Box(
                left=left, top=top, right=right, bottom=bottom,
                confidence=min(max(float(item["confidence"]), 0.0), 1.0),
                class_id=int(item.get("class_id", 0)),
                class_name=str(item.get("class_name") or f"class_{int(item.get('class_id', 0))}"),
            ))
        stages = dict(output.get("stages_ms", {}))
        stages["cloud_normalize_ms"] = (time.perf_counter() - normalize_started) * 1000
        return boxes, elapsed_ms, stages
