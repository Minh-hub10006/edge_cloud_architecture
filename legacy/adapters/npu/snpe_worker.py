"""Persistent SNPE worker process — loads DLC once, serves infer via Pipe."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import traceback
from multiprocessing.connection import Connection
from typing import Dict, Optional, Tuple

import numpy as np

from legacy.adapters.npu.snpe_env import snpe_persistent_enabled


class SnpeWorkerClient:
    """Parent-side client for a long-lived SNPE worker."""

    def __init__(
        self,
        dlc_path: str,
        *,
        runtime: str = "dsp",
        perf_profile: str = "burst",
        storage_dir: Optional[str] = None,
    ):
        self.dlc_path = dlc_path
        self.runtime = runtime
        self.perf_profile = perf_profile
        self.storage_dir = storage_dir
        self._conn: Optional[Connection] = None
        self._proc: Optional[mp.Process] = None
        self._lock = threading.Lock()
        self._start_worker()

    def _start_worker(self) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_worker_main,
            args=(
                self.dlc_path,
                self.runtime,
                self.perf_profile,
                self.storage_dir,
                child_conn,
            ),
            daemon=True,
            name=f"snpe-worker-{os.path.basename(self.dlc_path)}",
        )
        proc.start()
        child_conn.close()
        try:
            msg = parent_conn.recv()
        except EOFError as exc:
            proc.join(timeout=2)
            raise RuntimeError("SNPE worker died during startup") from exc
        if not isinstance(msg, dict) or not msg.get("ok"):
            raise RuntimeError(msg.get("error", "SNPE worker startup failed"))
        self._conn = parent_conn
        self._proc = proc

    def infer_raw(self, input_tensor: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        if self._conn is None:
            raise RuntimeError("SNPE worker not running")
        arr = np.ascontiguousarray(input_tensor, dtype=np.float32)
        with self._lock:
            if self._proc is not None and not self._proc.is_alive():
                raise RuntimeError("SNPE worker process exited")
            self._conn.send({"op": "infer", "shape": arr.shape, "dtype": "float32"})
            self._conn.send_bytes(arr.tobytes())
            reply = self._conn.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "infer failed"))
            outputs: Dict[str, np.ndarray] = {}
            for item in reply.get("outputs", []):
                raw = self._conn.recv_bytes()
                outputs[item["name"]] = np.frombuffer(raw, dtype=np.float32).reshape(item["shape"])
            return float(reply.get("dt_s", 0.0)), outputs

    def close(self) -> None:
        conn, proc = self._conn, self._proc
        self._conn, self._proc = None, None
        if conn is not None:
            try:
                conn.send({"op": "shutdown"})
                conn.close()
            except Exception:
                pass
        if proc is not None and proc.is_alive():
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_WORKERS: Dict[str, SnpeWorkerClient] = {}
_WORKERS_LOCK = threading.Lock()


def get_worker_client(
    dlc_path: str,
    *,
    runtime: str = "dsp",
    perf_profile: str = "burst",
    storage_dir: Optional[str] = None,
) -> SnpeWorkerClient:
    key = f"{dlc_path}|{runtime}|{perf_profile}|{storage_dir or ''}"
    with _WORKERS_LOCK:
        client = _WORKERS.get(key)
        if client is None:
            client = SnpeWorkerClient(
                dlc_path,
                runtime=runtime,
                perf_profile=perf_profile,
                storage_dir=storage_dir,
            )
            _WORKERS[key] = client
        return client


def try_get_worker_client(
    dlc_path: str,
    *,
    runtime: str = "dsp",
    perf_profile: str = "burst",
    storage_dir: Optional[str] = None,
) -> Optional[SnpeWorkerClient]:
    if not snpe_persistent_enabled():
        return None
    try:
        return get_worker_client(
            dlc_path,
            runtime=runtime,
            perf_profile=perf_profile,
            storage_dir=storage_dir,
        )
    except Exception as exc:
        print(f"[SnpeWorker] unavailable ({exc})")
        return None


def _worker_main(
    dlc_path: str,
    runtime: str,
    perf_profile: str,
    storage_dir: Optional[str],
    conn: Connection,
) -> None:
    backend = None
    try:
        from legacy.adapters.npu.snpe_python_backend import SnpePythonBackend

        backend = SnpePythonBackend(
            dlc_path,
            runtime=runtime,
            perf_profile=perf_profile,
            storage_dir=storage_dir,
        )
        conn.send({"ok": True, "backend": "snpe_python"})
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
        conn.close()
        return

    while True:
        try:
            if not conn.poll(timeout=300):
                break
            msg = conn.recv()
            op = msg.get("op")
            if op == "shutdown":
                break
            if op != "infer":
                conn.send({"ok": False, "error": f"unknown op {op!r}"})
                continue
            shape = tuple(int(x) for x in msg["shape"])
            nbytes = int(np.prod(shape)) * 4
            raw = conn.recv_bytes()
            if len(raw) != nbytes:
                conn.send({"ok": False, "error": f"payload size {len(raw)} != {nbytes}"})
                continue
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            t0 = time.perf_counter()
            dt_s, outputs = backend.infer_raw(arr)
            dt_s = time.perf_counter() - t0
            meta = []
            for name, tensor in outputs.items():
                flat = np.ascontiguousarray(tensor, dtype=np.float32)
                meta.append({"name": name, "shape": list(flat.shape)})
            conn.send({"ok": True, "dt_s": dt_s, "outputs": meta})
            for item in meta:
                conn.send_bytes(outputs[item["name"]].astype(np.float32, copy=False).tobytes())
        except EOFError:
            break
        except Exception:
            try:
                conn.send({"ok": False, "error": traceback.format_exc()})
            except Exception:
                break
    try:
        if backend is not None:
            backend.close()
    except Exception:
        pass
    conn.close()
