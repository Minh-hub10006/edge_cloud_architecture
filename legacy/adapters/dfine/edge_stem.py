"""Edge stem cut encoder — QNN HTP → INT8/int4gz uplink."""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np

from detectors.dfine.stem_codec import mock_stem, pack_stem, STEM_SHAPE
from detectors.npu.qnn_stem_backend import try_create_stem_backend


def _stem_direct_enabled() -> bool:
    raw = os.environ.get("STEM_DIRECT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _stem_warmup_frames() -> int:
    return max(0, int(os.environ.get("STEM_WARMUP", "100")))


class EdgeStem:
    """Runs edge_stem cut on NPU (direct or persistent worker) or ORT fallback."""

    def __init__(
        self,
        context_bin: Optional[str] = None,
        onnx_path: Optional[str] = None,
    ):
        self.context_bin = context_bin
        self.onnx_path = onnx_path
        self.runtime = "mock"
        self.infer_mode = "mock"
        self._worker = None
        self._direct = None

        if _stem_direct_enabled():
            created = try_create_stem_backend(context_bin=context_bin, onnx_path=onnx_path)
            if created is not None:
                self._direct, mode = created
                self.runtime = mode
                self.infer_mode = mode
                print(f"[EdgeStem] direct in-process backend mode={mode}")
                self._warmup()
                return

        from detectors.npu.qnn_stem_worker import qnn_stem_persistent_enabled, try_get_qnn_stem_worker

        if qnn_stem_persistent_enabled():
            worker = try_get_qnn_stem_worker(context_bin=context_bin, onnx_path=onnx_path)
            if worker is not None:
                self._worker = worker
                self.runtime = f"qnn_htp_{worker.mode}"
                self.infer_mode = worker.mode
                print(f"[EdgeStem] QNN persistent worker mode={worker.mode}")
                self._warmup()
                return

        created = try_create_stem_backend(context_bin=context_bin, onnx_path=onnx_path)
        if created is not None:
            self._direct, mode = created
            self.runtime = mode
            self.infer_mode = mode
            print(f"[EdgeStem] direct backend mode={mode}")
            self._warmup()
            return

        print(
            "[EdgeStem] NPU/ORT unavailable — mock stem. "
            "Set STEM_QNN_BIN or EDGE_STEM_ONNX."
        )

    def _warmup(self) -> None:
        n = _stem_warmup_frames()
        if n <= 0:
            return
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(n):
            self.forward(dummy)
        dt = time.perf_counter() - t0
        print(f"[EdgeStem] warmup {n} frames in {dt:.2f}s ({n/max(dt,1e-6):.1f} fps)")

    def forward(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        if self._direct is not None:
            return self._direct.infer_stem(frame_bgr)
        if self._worker is not None:
            return self._worker.infer_stem(frame_bgr)

        t0 = time.time()
        time.sleep(0.009)
        stem, _ = mock_stem()
        return time.time() - t0, stem

    def encode(self, frame_bgr: np.ndarray) -> Tuple[float, bytes]:
        t, stem = self.forward(frame_bgr)
        if tuple(stem.shape) != STEM_SHAPE:
            stem = np.asarray(stem, dtype=np.float32).reshape(STEM_SHAPE)
        payload, _layout = pack_stem(stem)
        return t, payload
