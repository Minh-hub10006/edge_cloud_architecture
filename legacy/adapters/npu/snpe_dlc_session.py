"""SNPE DLC inference — persistent session (Python/worker) with subprocess fallback."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from legacy.adapters.npu.snpe_env import setup_qairt_env, snpe_persistent_enabled

InferMode = Literal["snpe_daemon", "snpe_python", "snpe_worker", "snpe_subprocess"]


def _snpe_bin() -> str:
    explicit = os.environ.get("SNPE_ITENSOR_BIN", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    cand = os.path.expanduser("~/edge_artifacts/bin/snpe-itensor")
    if os.path.isfile(cand):
        return cand
    qairt = os.environ.get("QAIRT_ROOT", os.path.expanduser("~/sigm/qairt/2.43.0.260128"))
    return os.path.join(qairt, "examples/SNPE/NativeCpp/SampleCode_CAPI/ITensor/snpe-itensor")


def resolve_dlc(name: str, explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    env_key = f"{name.upper()}_DLC"
    p = os.environ.get(env_key, "").strip()
    if p and os.path.isfile(p):
        return p
    art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
    for sub in ("npu243", "yolo243_snpe", "npu"):
        cand = os.path.join(art, sub, f"{name}_int8.dlc")
        if os.path.isfile(cand):
            return cand
    return os.path.join(art, "npu243", f"{name}_int8.dlc")


class SnpeDlcSession:
    """Persistent SNPE DLC session when possible; subprocess only as last resort."""

    def __init__(
        self,
        dlc_path: str,
        *,
        runtime: str = "dsp",
        perf_profile: str = "burst",
        storage_dir: Optional[str] = None,
        require_daemon: bool = False,
        output_tensors: Optional[tuple[str, ...]] = None,
        input_bytes: Optional[int] = None,
        graph_name: Optional[str] = None,
    ):
        self.dlc_path = dlc_path
        self.runtime = runtime
        self.perf_profile = perf_profile
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(dlc_path), ".snpe_cache"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        if not os.path.isfile(self.dlc_path):
            raise FileNotFoundError(self.dlc_path)

        self.mode: InferMode = "snpe_subprocess"
        self._python_backend = None
        self._worker_client = None
        self._daemon_client = None
        self._infer_lock = __import__("threading").Lock()
        self._bin: Optional[str] = None

        if require_daemon:
            if runtime != "dsp":
                raise ValueError("strict daemon requires DSP runtime")
            from legacy.adapters.npu.snpe_daemon_manager import try_get_daemon_client
            self._daemon_client = try_get_daemon_client(
                dlc_path, runtime=runtime, perf_profile=perf_profile,
                output_tensors=output_tensors, input_bytes=input_bytes, graph_name=graph_name)
            if self._daemon_client is None:
                raise RuntimeError("Persistent SNPE DSP daemon unavailable; no fallback permitted")
            self.mode = "snpe_daemon"
        elif snpe_persistent_enabled():
            self._try_persistent_backends()
        if self.mode == "snpe_subprocess":
            self._bin = _snpe_bin()
            if not os.path.isfile(self._bin):
                raise FileNotFoundError(
                    f"snpe-itensor not found ({self._bin}). "
                    "Run scripts/build_snpe_itensor_edge.sh or enable SNPE Python bindings."
                )

    def _try_persistent_backends(self) -> None:
        prefer = os.environ.get("SNPE_PERSISTENT_MODE", "auto").strip().lower()
        if prefer in ("auto", "daemon", "native"):
            from legacy.adapters.npu.snpe_daemon_manager import try_get_daemon_client

            client = try_get_daemon_client(
                self.dlc_path,
                runtime=self.runtime,
                perf_profile=self.perf_profile,
            )
            if client is not None:
                self._daemon_client = client
                self.mode = "snpe_daemon"
                print(f"[SnpeDlc] native daemon: {self.dlc_path}")
                return
            if prefer in ("daemon", "native"):
                return

        if prefer in ("auto", "python", "inprocess"):
            from legacy.adapters.npu.snpe_python_backend import try_create_python_backend

            backend = try_create_python_backend(
                self.dlc_path,
                runtime=self.runtime,
                perf_profile=self.perf_profile,
                storage_dir=self.storage_dir,
            )
            if backend is not None:
                self._python_backend = backend
                self.mode = "snpe_python"
                print(f"[SnpeDlc] persistent in-process: {self.dlc_path}")
                return
            if prefer == "python":
                return

        if prefer in ("auto", "worker"):
            from legacy.adapters.npu.snpe_worker import try_get_worker_client

            client = try_get_worker_client(
                self.dlc_path,
                runtime=self.runtime,
                perf_profile=self.perf_profile,
                storage_dir=self.storage_dir,
            )
            if client is not None:
                self._worker_client = client
                self.mode = "snpe_worker"
                print(f"[SnpeDlc] persistent worker: {self.dlc_path}")
                return

    def infer_raw(self, input_tensor: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        if self._daemon_client is not None:
            with self._infer_lock:
                return self._daemon_client.infer_raw(input_tensor)
        if self._python_backend is not None:
            return self._python_backend.infer_raw(input_tensor)
        if self._worker_client is not None:
            return self._worker_client.infer_raw(input_tensor)
        return self._infer_subprocess(input_tensor)

    def _infer_subprocess(self, input_tensor: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="snpe_dlc_") as tmp:
            tmp_p = Path(tmp)
            in_raw = tmp_p / "in.raw"
            input_list = tmp_p / "input_list.txt"
            out_dir = tmp_p / "out"
            out_dir.mkdir()
            in_raw.write_bytes(np.ascontiguousarray(input_tensor, dtype=np.float32).tobytes())
            input_list.write_text(f"{in_raw}\n", encoding="utf-8")
            cmd = [
                self._bin or _snpe_bin(),
                "-d",
                self.dlc_path,
                "-i",
                str(input_list),
                "-o",
                str(out_dir),
                "-r",
                self.runtime,
                "-p",
                self.perf_profile,
                "-c",
            ]
            subprocess.run(
                cmd,
                check=True,
                env=setup_qairt_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            outputs: Dict[str, np.ndarray] = {}
            for f in sorted(out_dir.glob("*.raw")):
                outputs[f.stem] = np.fromfile(f, dtype=np.float32)
        return time.perf_counter() - t0, outputs

    def close(self) -> None:
        if self._python_backend is not None:
            try:
                self._python_backend.close()
            except Exception:
                pass
            self._python_backend = None
        # Worker clients are process-singleton; do not close per session.

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
