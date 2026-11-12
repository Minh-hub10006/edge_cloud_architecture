"""RTMDet-Small NPU lane. No YOLO decoder or CPU fallback."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .model_artifact import load_artifact
from .detection_decode import rtmdet_decode


@dataclass(frozen=True)
class PreparedRtmFrame:
    tensor: np.ndarray
    width: int
    height: int
    scale_x: float
    scale_y: float
    stages_ms: dict


class RtmdetSmallSnpeAdapter:
    def __init__(self, dlc_path: str, *, manifest_path: str | None = None,
                 confidence: float = .10, top_k: int = 300):
        if not 0 < confidence <= 1 or type(top_k) is not int or not 1 <= top_k <= 300:
            raise ValueError("RTMDet confidence must be >0 and top_k 1..300")
        self.artifact_path, self.manifest = load_artifact(dlc_path, "rtmdet_small", manifest_path)
        self.model_id = self.manifest["model_id"]
        self.confidence, self.top_k = confidence, top_k
        if self.manifest["output_shapes"] != {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]}:
            raise ValueError("RTMDet 640 export requires 8400 decoded anchors")
        from legacy.adapters.npu.snpe_dlc_session import SnpeDlcSession
        self.session = SnpeDlcSession(str(self.artifact_path), runtime="dsp")
        if self.session.runtime != "dsp" or self.session.mode == "snpe_subprocess":
            raise RuntimeError("RTMDet benchmark requires persistent SNPE DSP runtime; no subprocess fallback")
        self.runtime = "snpe_dsp"

    @property
    def settings(self):
        return {"confidence": self.confidence, "top_k": self.top_k, "nms_iou": .65,
                "input_layout": self.manifest["input_layout"], "input_size": 640,
                "artifact_contract": self.manifest}

    def prepare_frame(self, bgr):
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
            raise ValueError("RTMDet input must be nonempty uint8 HxWx3 BGR")
        import cv2
        started = perf_counter()
        height, width = bgr.shape[:2]
        ratio = min(640 / width, 640 / height)
        new_width, new_height = int(width * ratio + .5), int(height * ratio + .5)
        # MMDetection keep-ratio resize, padding on right/bottom (not centered).
        image = np.full((640, 640, 3), 114, dtype=np.float32)
        image[:new_height, :new_width] = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        image = (image - np.array([103.53, 116.28, 123.675], dtype=np.float32)) / np.array([57.375, 57.12, 58.395], dtype=np.float32)
        tensor = image[None] if self.manifest["input_layout"] == "nhwc" else image.transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(tensor)  # owned, no mutable model scratch
        return PreparedRtmFrame(tensor, width, height, new_width / width, new_height / height,
            {"edge_preprocess_ms": (perf_counter() - started) * 1000})

    def infer_prepared(self, prepared):
        stages = dict(prepared.stages_ms)
        started = perf_counter()
        reported, outputs = self.session.infer_raw(prepared.tensor)
        inferred = perf_counter()
        # SNPE native runners often flatten tensors; shapes are manifest-owned.
        if set(outputs) != {"boxes", "scores"}:
            raise ValueError("SNPE RTMDet output names mismatch; inspect DLC, do not guess")
        outputs = {name: np.asarray(value).reshape(self.manifest["output_shapes"][name]) for name, value in outputs.items()}
        boxes = rtmdet_decode(outputs, width=prepared.width, height=prepared.height,
            scale_x=prepared.scale_x, scale_y=prepared.scale_y,
            confidence=self.confidence, top_k=self.top_k)
        stages.update(edge_session_call_ms=(inferred - started) * 1000,
                      edge_npu_reported_ms=float(reported) * 1000,
                      edge_postprocess_ms=(perf_counter() - inferred) * 1000)
        elapsed = sum(value for key, value in stages.items() if key != "edge_npu_reported_ms")
        return boxes, elapsed, stages

    def infer(self, bgr):
        boxes, elapsed, _ = self.infer_prepared(self.prepare_frame(bgr))
        return boxes, elapsed
