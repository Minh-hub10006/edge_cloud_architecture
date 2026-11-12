"""Pinned split-v8 YOLO11s320 candidate on the shared Edge inference path."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np

from contracts import Box
import yolo11s_contract
from yolo11s_contract import SHAPES, Prepared, decode, digest, prepare

MODEL_ID = "yolo11s320_w8a8_qcs6490"
GRAPH_NAME = "yolo11s320_float"
DLC_SHA256 = "1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3"
# Same weights re-prepared on Cloud249 with a QCS6490 cache for boxes+scores (Edge252 cannot
# prepare HTP graphs on device).
DLC_SHA256_ACCEPTED = {DLC_SHA256, "4e1e5246d4e027bf66fd82a52e0a59e5cd23654fbbc2f3f8a4b3c5f7e6850316"}
# GPU Edges run the split-v8 float ONNX the DLC was converted from (NCHW input, same decode).
ONNX_MODEL_ID = "yolo11s320_float_ort"
ONNX_SHA256 = "62ff9e6d7803d30b741d3933845bc23e4404a2e4d37273db540dd6248c1300db"


@dataclass(frozen=True)
class PreparedYolo11Frame:
    tensor: np.ndarray
    geometry: Prepared
    stages_ms: dict


class Yolo11sSnpeAdapter:
    def __init__(self, dlc_path, *, allow_diagnostic=False, confidence=.1, nms_iou=.5):
        if not allow_diagnostic:
            raise ValueError("YOLO11s parity FAIL candidate requires explicit diagnostic opt-in")
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid YOLO11 confidence")
        if not np.isfinite(nms_iou) or not 0 < nms_iou <= 1:
            raise ValueError("invalid YOLO11 NMS")
        self.artifact_path = Path(dlc_path).expanduser().resolve(strict=True)
        self.artifact_sha256 = digest(self.artifact_path)
        self.confidence, self.nms_iou = confidence, nms_iou
        if self.artifact_path.suffix.lower() == ".onnx":
            if self.artifact_sha256 != ONNX_SHA256:
                raise ValueError("YOLO11s split-v8 ONNX checksum mismatch")
            from .ort_session import OrtCudaSession
            self.session = OrtCudaSession(str(self.artifact_path))
            self.model_id, self.runtime, self.input_layout = ONNX_MODEL_ID, "cuda", "nchw"
            return
        if self.artifact_sha256 not in DLC_SHA256_ACCEPTED:
            raise ValueError("YOLO11s split-v8 QCS6490 DLC checksum mismatch")
        from legacy.adapters.npu.snpe_dlc_session import SnpeDlcSession
        self.session = SnpeDlcSession(str(self.artifact_path), runtime="dsp",
            require_daemon=True, output_tensors=("boxes", "scores"), input_bytes=320 * 320 * 3 * 4,
            graph_name=GRAPH_NAME)
        if self.session.runtime != "dsp" or self.session.mode != "snpe_daemon":
            self.session.close()
            raise RuntimeError("YOLO11s requires persistent DSP daemon, no CPU/subprocess fallback")
        self.model_id = MODEL_ID
        self.runtime = "snpe_dsp"
        self.input_layout = "nhwc"

    @property
    def result_attributes(self):
        return {"diagnostic_only": True, "production_ready": False,
                "quality_status": "NOT_EVALUATED", "parity_gate": "FAIL",
                "artifact_sha256": self.artifact_sha256, "runtime_requested": self.session.runtime,
                "session_mode": self.session.mode, "graph_name": GRAPH_NAME,
                "daemon_contract": "SNPE_DAEMON_GRAPH_DSP_V3" if self.runtime == "snpe_dsp" else None,
                "init_cache": False}

    @property
    def settings(self):
        return {**self.result_attributes, "confidence": self.confidence, "nms_iou": self.nms_iou,
                "nms_impl": "numpy-matrix-v3" if yolo11s_contract.NMS_IMPL == "matrix" else "python-greedy-v1",
                "input_layout": self.input_layout, "input_size": 320, "output_shapes": SHAPES}

    def prepare_frame(self, bgr):
        started = perf_counter()
        geometry = prepare(bgr)
        tensor = (geometry.tensor if self.input_layout == "nchw"
                  else np.ascontiguousarray(geometry.tensor.transpose(0, 2, 3, 1)))
        return PreparedYolo11Frame(tensor, geometry,
                                  {"edge_preprocess_ms": (perf_counter() - started) * 1000})

    def infer_prepared(self, prepared):
        stages = dict(prepared.stages_ms)
        started = perf_counter()
        reported, outputs = self.session.infer_raw(prepared.tensor)
        stages["edge_session_call_ms"] = (perf_counter() - started) * 1000
        stages["edge_npu_reported_ms"] = float(reported) * 1000
        if not np.isfinite(reported) or reported < 0:
            raise ValueError("invalid daemon timing")
        started = perf_counter()
        if set(outputs) != set(SHAPES):
            raise ValueError("YOLO11 daemon must return boxes and scores by name")
        shaped = {}
        for name, shape in SHAPES.items():
            values = np.asarray(outputs[name])
            if values.dtype != np.float32 or values.size != int(np.prod(shape)):
                raise ValueError("YOLO11 daemon tensor dtype/size mismatch: " + name)
            if values.shape not in {(values.size,), shape}:
                raise ValueError("YOLO11 daemon tensor layout mismatch: " + name)
            shaped[name] = values.reshape(shape)
        detections = decode(shaped, prepared.geometry, nms_iou=self.nms_iou, confidence=self.confidence)
        boxes = [Box(d.left, d.top, d.right, d.bottom, d.confidence, d.class_id, d.class_name)
                 for d in detections]
        stages["edge_postprocess_ms"] = (perf_counter() - started) * 1000
        return boxes, sum(v for k, v in stages.items() if k != "edge_npu_reported_ms"), stages

    def infer(self, bgr):
        boxes, elapsed, _ = self.infer_prepared(self.prepare_frame(bgr))
        return boxes, elapsed
