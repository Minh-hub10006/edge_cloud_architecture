"""ONNX Runtime CUDA stand-in for SnpeDlcSession on GPU Edges (same infer_raw contract)."""
from __future__ import annotations

from time import perf_counter

import numpy as np


class OrtCudaSession:
    runtime = "cuda"
    mode = "ort_cuda"

    def __init__(self, onnx_path: str, *, output_names: tuple[str, ...] = ("boxes", "scores")) -> None:
        import onnxruntime as ort
        try:
            ort.preload_dlls()  # CUDA/cuDNN DLLs from the nvidia-* / torch wheels (Windows).
        except (AttributeError, OSError):
            pass
        self.session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        # Refuse a silent CPU fallback: Edge GPU numbers must be GPU numbers.
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDAExecutionProvider unavailable for {onnx_path}")
        names = [o.name for o in self.session.get_outputs()]
        if sorted(names) != sorted(output_names):
            raise ValueError(f"ONNX outputs {names} != {list(output_names)}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = names

    def infer_raw(self, input_tensor: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        started = perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return perf_counter() - started, dict(zip(self.output_names, outputs))

    def get_providers(self) -> list[str]:
        return self.session.get_providers()

    def close(self) -> None:
        self.session = None
