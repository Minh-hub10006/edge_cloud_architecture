"""Path B Edge half (v4 step 4): RT-DETRv2-S up to layer2 on the QCS6490 NPU -> packed cut tensor for the ticket.

DLC = head_w8a8_pc (per-channel W8A8, gates G2/G3 passed 5/11), run by the persistent graph-scoped SNPE daemon
(the YOLO11s v3 daemon, YOLO11S_SNPE_DAEMON_BIN), input NHWC float32 640 prepared exactly like the Cloud model
(inference/cloud_detr.py: INTER_AREA, RGB, /255). The Cloud tail (inference/triton_detr.RtdetrSplitAdapter) only
accepts tensors whose head id matches RTDETR_SPLIT_HEAD_ID.
"""
from __future__ import annotations

from pathlib import Path
import os
import threading
from time import perf_counter

import numpy as np

from contracts.cut_tensor import pack

# 14/11: 2 bits per value, one range per channel (CUT2): ~90 KB instead of ~450 KB, F1 0.839 vs 0.8445 on UA-DETRAC
# (scripts/compress_survey.py). V4_CUT_BITS=8 sends the old CUT1 uint8 tensor.
CUT_BITS = int(os.getenv("V4_CUT_BITS", "2"))

HEAD_ID = "rtl2-w8a8pc"
DLC_SHA256 = "2e0c4b09df237165efb70e22e0ca47ff0845b8af58d7dd8c6290103589a5f673"
GRAPH_NAME = "head_fp32"  # graph name inside the DLC (converted from head_fp32.dlc on 249)
OUTPUT = "/model/backbone/res_layers.1/blocks.1/act/Relu_output_0"
CUT_SHAPE_NHWC = (1, 80, 80, 128)


def prepare(bgr):
    import cv2
    small = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[None], dtype=np.float32) / 255


class RtdetrHeadSnpe:
    def __init__(self, dlc_path, *, session=None):
        self.dlc_path = Path(dlc_path).expanduser().resolve(strict=True)
        if session is None:
            from yolo11s_contract import digest
            if digest(self.dlc_path) != DLC_SHA256:
                raise ValueError("RT-DETR head DLC checksum mismatch (expected head_w8a8_pc)")
            from legacy.adapters.npu.snpe_dlc_session import SnpeDlcSession
            session = SnpeDlcSession(str(self.dlc_path), runtime="dsp", require_daemon=True, output_tensors=(OUTPUT,),
                                     input_bytes=640 * 640 * 3 * 4, graph_name=GRAPH_NAME)
            if session.mode != "snpe_daemon":
                session.close()
                raise RuntimeError("RT-DETR head requires the persistent DSP daemon")
        self.session, self.head_id = session, HEAD_ID
        self._npu = threading.Lock()

    def close(self):
        """Stops this process's daemon (SnpeDlcSession.close does not): a leaked one keeps its DSP session and later
        daemons fail with "DSP unavailable". shortcut: a killed agent still leaks it; scripts must clean up after kills."""
        from legacy.adapters.npu.snpe_daemon_manager import _DAEMONS
        for handle in list(_DAEMONS.values()):
            if handle.dlc_path == str(self.dlc_path):
                handle.close()  # terminates the process this adapter started

    def encode(self, bgr) -> tuple[bytes, dict]:
        """BGR frame -> (packed cut tensor, stage ms)."""
        t0 = perf_counter()
        x = prepare(bgr)
        t1 = perf_counter()
        with self._npu:  # the daemon's output is a shared-memory view: the next frame overwrites it, so copy under the lock
            _, outs = self.session.infer_raw(x)
            nhwc = np.array(outs[OUTPUT], dtype=np.float32).reshape(CUT_SHAPE_NHWC)
        t2 = perf_counter()
        data = pack(np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2)), width=bgr.shape[1], height=bgr.shape[0],
                    head_id=self.head_id, bits=CUT_BITS)
        t3 = perf_counter()
        return data, {"prepare_ms": (t1 - t0) * 1000, "npu_ms": (t2 - t1) * 1000, "pack_ms": (t3 - t2) * 1000}


ONNX_HEAD_ID = "rtl2-onnx-fp32"
ONNX_SHA256 = "5e72f7c91386fe9e7a94282fffff9e5edd650064befa6e026b3db3e22ba5486f"  # head.onnx cut from rtdetrv2_s_640.onnx (scripts/cut_points.py, 4/11)


class RtdetrHeadOnnx:
    """Path B on a GPU Edge (247): the float head on onnxruntime CUDA, same preprocessing and wire format."""

    def __init__(self, onnx_path, *, session=None):
        self.onnx_path = Path(onnx_path).expanduser().resolve(strict=True)
        if session is None:
            from yolo11s_contract import digest
            if digest(self.onnx_path) != ONNX_SHA256:
                raise ValueError("RT-DETR head ONNX checksum mismatch (expected the 4/11 layer2 cut)")
            from .ort_session import OrtCudaSession
            session = OrtCudaSession(str(self.onnx_path), output_names=(OUTPUT,))
        self.session, self.head_id = session, ONNX_HEAD_ID

    def close(self):
        pass

    def encode(self, bgr) -> tuple[bytes, dict]:
        t0 = perf_counter()
        x = np.ascontiguousarray(prepare(bgr).transpose(0, 3, 1, 2))
        t1 = perf_counter()
        _, outs = self.session.infer_raw(x)
        t2 = perf_counter()
        data = pack(np.ascontiguousarray(outs[OUTPUT], dtype=np.float32), width=bgr.shape[1], height=bgr.shape[0],
                    head_id=self.head_id, bits=CUT_BITS)
        t3 = perf_counter()
        return data, {"prepare_ms": (t1 - t0) * 1000, "npu_ms": (t2 - t1) * 1000, "pack_ms": (t3 - t2) * 1000}


def open_head(path):
    """.onnx = GPU Edge (onnxruntime CUDA), anything else = NPU DLC (persistent SNPE daemon)."""
    return RtdetrHeadOnnx(path) if str(path).lower().endswith(".onnx") else RtdetrHeadSnpe(path)
