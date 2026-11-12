"""In-process SNPE DLC inference via QAIRT Python bindings (persistent session)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from legacy.adapters.npu.snpe_env import apply_qairt_to_process


class SnpePythonBackend:
    """Load DLC once; execute many times without subprocess spawn."""

    def __init__(
        self,
        dlc_path: str,
        *,
        runtime: str = "dsp",
        perf_profile: str = "burst",
        storage_dir: Optional[str] = None,
    ):
        apply_qairt_to_process()
        self.dlc_path = dlc_path
        self.runtime = runtime
        self.perf_profile = perf_profile
        self.storage_dir = storage_dir
        self._snpe: Any = None
        self._input_name: Optional[str] = None
        self._output_names: list[str] = []
        self._load()

    def _load(self) -> None:
        builder = _build_snpe_builder(
            self.dlc_path,
            runtime=self.runtime,
            perf_profile=self.perf_profile,
            storage_dir=self.storage_dir,
        )
        self._snpe = builder
        names = _tensor_map_names(self._snpe, "input")
        if not names:
            raise RuntimeError("SNPE model has no input tensors")
        self._input_name = names[0]
        self._output_names = _tensor_map_names(self._snpe, "output")

    def infer_raw(self, input_tensor: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        if self._snpe is None or self._input_name is None:
            raise RuntimeError("SNPE backend not loaded")
        t0 = time.perf_counter()
        arr = np.ascontiguousarray(input_tensor, dtype=np.float32)
        _set_input_tensor(self._snpe, self._input_name, arr)
        raw_out = _execute(self._snpe)
        outputs = _normalize_outputs(raw_out, self._output_names)
        return time.perf_counter() - t0, outputs

    def close(self) -> None:
        self._snpe = None


def _build_snpe_builder(
    dlc_path: str,
    *,
    runtime: str,
    perf_profile: str,
    storage_dir: Optional[str],
) -> Any:
    from snpe.snpe_builder import SnpeBuilder

    builder = SnpeBuilder(dlc_path)
    rt = _runtime_order(SnpeBuilder, runtime)
    if rt:
        builder.set_runtime_processor_order(rt)
    prof = _perf_profile(SnpeBuilder, perf_profile)
    if prof is not None:
        try:
            builder.set_perf_profile(prof)
        except Exception:
            pass
    if storage_dir:
        for meth in ("set_storage_directory", "set_output_directory"):
            fn = getattr(builder, meth, None)
            if callable(fn):
                try:
                    fn(storage_dir)
                except Exception:
                    pass
    try:
        builder.set_use_user_supplied_buffers(False)
    except Exception:
        pass
    try:
        builder.set_init_cache_enabled(True)
    except Exception:
        pass
    return builder.build()


def _runtime_order(SnpeBuilder: Any, runtime: str) -> list[Any]:
    Runtime = SnpeBuilder.Runtime
    key = runtime.strip().lower()
    mapping = {
        "dsp": [Runtime.DSP, Runtime.CPU],
        "htp": [Runtime.DSP, Runtime.CPU],
        "gpu": [Runtime.GPU, Runtime.CPU],
        "cpu": [Runtime.CPU],
    }
    return mapping.get(key, mapping["dsp"])


def _perf_profile(SnpeBuilder: Any, name: str) -> Any:
    Perf = getattr(SnpeBuilder, "PerfProfile", None) or getattr(SnpeBuilder, "PerformanceProfile", None)
    if Perf is None:
        return None
    key = name.strip().lower()
    for attr in (key.upper(), key.capitalize(), key):
        if hasattr(Perf, attr):
            return getattr(Perf, attr)
    return getattr(Perf, "BURST", None)


def _tensor_map_names(snpe: Any, kind: str) -> list[str]:
    if kind == "input":
        getter = getattr(snpe, "get_input_tensors", None)
    else:
        getter = getattr(snpe, "get_output_tensors", None)
    if not callable(getter):
        return []
    tensors = getter()
    if isinstance(tensors, dict):
        return list(tensors.keys())
    if isinstance(tensors, (list, tuple)):
        out: list[str] = []
        for t in tensors:
            name = getattr(t, "name", None) or getattr(t, "tensor_name", None)
            out.append(str(name) if name else f"{kind}_{len(out)}")
        return out
    return []


def _set_input_tensor(snpe: Any, name: str, arr: np.ndarray) -> None:
    tensors = snpe.get_input_tensors()
    if isinstance(tensors, dict):
        tensor = tensors[name]
    else:
        tensor = tensors[0]
    if hasattr(tensor, "load"):
        tensor.load(arr)
        return
    if hasattr(tensor, "set_tensor_data"):
        tensor.set_tensor_data(arr)
        return
    # Last resort: in-place assignment on buffer views
    view = np.asarray(tensor)
    if view.shape == arr.shape:
        view[:] = arr
        return
    raise RuntimeError(f"Cannot load input tensor {name!r} shape {arr.shape}")


def _execute(snpe: Any) -> Any:
    inputs = snpe.get_input_tensors()
    if hasattr(snpe, "execute"):
        return snpe.execute(inputs)
    if hasattr(snpe, "executeITensors"):
        return snpe.executeITensors(inputs)
    raise RuntimeError("SNPE instance has no execute method")


def _normalize_outputs(raw: Any, names: list[str]) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            outputs[str(k)] = _as_numpy(v)
        return outputs
    if isinstance(raw, (list, tuple)):
        for i, v in enumerate(raw):
            key = names[i] if i < len(names) else f"output_{i}"
            outputs[key] = _as_numpy(v)
        return outputs
    if len(names) == 1:
        outputs[names[0]] = _as_numpy(raw)
        return outputs
    raise RuntimeError("Unexpected SNPE output structure")


def _as_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32, copy=False)
    if hasattr(tensor, "to_numpy"):
        return np.asarray(tensor.to_numpy(), dtype=np.float32)
    if hasattr(tensor, "data"):
        return np.asarray(tensor.data(), dtype=np.float32)
    return np.asarray(tensor, dtype=np.float32)


def try_create_python_backend(
    dlc_path: str,
    *,
    runtime: str = "dsp",
    perf_profile: str = "burst",
    storage_dir: Optional[str] = None,
) -> Optional[SnpePythonBackend]:
    try:
        return SnpePythonBackend(
            dlc_path,
            runtime=runtime,
            perf_profile=perf_profile,
            storage_dir=storage_dir,
        )
    except Exception as exc:
        print(f"[SnpePython] unavailable ({exc}); will try worker/subprocess fallback")
        return None
