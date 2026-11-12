"""Explicit RTMDet-S Float32 CPU fallback with the verified MMDetection contract."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .detection_decode import rtmdet_decode
from .model_artifact import load_artifact
from .rtmdet_snpe import PreparedRtmFrame


class RtmdetSmallFloatCpuAdapter:
    def __init__(self, onnx_path: str, *, threads: int = 2, confidence: float = .1,
                 top_k: int = 300, manifest_path: str | None = None):
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError("CPU threads must be 1..16")
        if not 0 < confidence <= 1 or type(top_k) is not int or not 1 <= top_k <= 300:
            raise ValueError("RTMDet confidence/top_k out of range")
        self.artifact_path, self.manifest = load_artifact(
            onnx_path, "rtmdet_small", manifest_path, rtmdet_float_onnx=True)
        expected = {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]}
        evidence = self.manifest.get("export_evidence", {})
        if (self.manifest["input_layout"] != "nchw" or self.manifest["output_shapes"] != expected
                or evidence.get("preprocess_vs_mmdetection") != "PASS"
                or evidence.get("torch_onnx_raw_parity") != "PASS"
                or evidence.get("onnx_decoder_vs_mmdetection", {}).get("gate") != "PASS"
                or evidence.get("input_mean") != [103.53, 116.28, 123.675]
                or evidence.get("input_std") != [57.375, 57.12, 58.395]
                or evidence.get("nms_iou") != .65):
            raise ValueError("Float CPU adapter requires the verified RTMDet rebuild contract")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(self.artifact_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Explicit Float CPU session did not select CPU exclusively")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or (inputs[0].name, inputs[0].type, inputs[0].shape) != (
                "input", "tensor(float)", [1, 3, 640, 640]):
            raise ValueError("Float CPU input signature mismatch")
        outputs = self.session.get_outputs()
        if len(outputs) != 2 or {v.name: v.shape for v in outputs} != expected or any(
                v.type != "tensor(float)" for v in outputs):
            raise ValueError("Float CPU output signature mismatch")
        self.model_id = self.manifest["model_id"] + "_float_cpu"
        self.runtime = "onnx_cpu"
        self.threads, self.confidence, self.top_k = threads, confidence, top_k

    @property
    def settings(self):
        return {"precision": "float32", "runtime": self.runtime, "cpu_threads": self.threads,
                "confidence": self.confidence, "top_k": self.top_k, "nms_iou": .65,
                "input_layout": "nchw", "input_size": 640, "artifact_contract": self.manifest}

    def prepare_frame(self, bgr):
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
            raise ValueError("RTMDet input must be nonempty uint8 HxWx3 BGR")
        import cv2
        start = perf_counter()
        height, width = bgr.shape[:2]
        ratio = min(640 / width, 640 / height)
        new_w, new_h = int(width * ratio + .5), int(height * ratio + .5)
        image = np.full((640, 640, 3), 114, dtype=np.float32)
        image[:new_h, :new_w] = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        image = (image - np.array([103.53, 116.28, 123.675], dtype=np.float32)) / np.array(
            [57.375, 57.12, 58.395], dtype=np.float32)
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        return PreparedRtmFrame(tensor, width, height, new_w / width, new_h / height,
                                {"edge_preprocess_ms": (perf_counter() - start) * 1000})

    def infer_prepared(self, prepared):
        start = perf_counter()
        values = self.session.run(["boxes", "scores"], {"input": prepared.tensor})
        session_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        outputs = dict(zip(("boxes", "scores"), values))
        for name, shape in self.manifest["output_shapes"].items():
            value = outputs[name]
            if value.shape != tuple(shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Float CPU output tensor contract mismatch: " + name)
        # The pinned test pipeline's Pad sets upstream img_shape to (640,640).
        # Clip to that shape BEFORE NMS, then inverse-scale/clip to the original image.
        boxes = outputs["boxes"].copy()
        boxes = np.clip(boxes, 0, 640)
        result = rtmdet_decode({"boxes": boxes, "scores": outputs["scores"]}, width=prepared.width,
                               height=prepared.height, scale_x=prepared.scale_x, scale_y=prepared.scale_y,
                               confidence=self.confidence, top_k=self.top_k, iou=.65)
        stages = {**prepared.stages_ms, "edge_session_call_ms": session_ms,
                  "edge_postprocess_ms": (perf_counter() - start) * 1000}
        return result, sum(stages.values()), stages

    def infer(self, bgr):
        boxes, elapsed, _ = self.infer_prepared(self.prepare_frame(bgr))
        return boxes, elapsed
