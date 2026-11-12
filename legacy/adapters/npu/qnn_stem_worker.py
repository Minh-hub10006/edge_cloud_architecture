"""Persistent QNN stem worker — avoids per-frame graph recompile on edge."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from multiprocessing.connection import Connection
from typing import Optional, Tuple

import numpy as np

from detectors.dfine.stem_codec import STEM_SHAPE
from detectors.npu.qnn_stem_backend import qnn_stem_persistent_enabled, try_create_stem_backend


class QnnStemWorkerClient:
    def __init__(self, *, context_bin: Optional[str] = None, onnx_path: Optional[str] = None):
        self.context_bin = context_bin
        self.onnx_path = onnx_path
        self.mode = "unknown"
        self._conn: Optional[Connection] = None
        self._proc: Optional[mp.Process] = None
        self._lock = threading.Lock()
        self._start_worker()

    def _start_worker(self) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_worker_main,
            args=(self.context_bin, self.onnx_path, child_conn),
            daemon=True,
            name="qnn-stem-worker",
        )
        proc.start()
        child_conn.close()
        msg = parent_conn.recv()
        if not isinstance(msg, dict) or not msg.get("ok"):
            raise RuntimeError(msg.get("error", "QNN stem worker startup failed"))
        self.mode = str(msg.get("mode", "qnn_stem_worker"))
        self._conn = parent_conn
        self._proc = proc

    def infer_stem(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        if self._conn is None:
            raise RuntimeError("QNN stem worker not running")
        arr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        with self._lock:
            if self._proc is not None and not self._proc.is_alive():
                raise RuntimeError("QNN stem worker process exited")
            self._conn.send({"op": "infer", "shape": list(arr.shape), "dtype": "uint8"})
            self._conn.send_bytes(arr.tobytes())
            reply = self._conn.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "infer failed"))
            raw = self._conn.recv_bytes()
            stem = np.frombuffer(raw, dtype=np.float32).reshape(reply.get("shape", STEM_SHAPE))
            return float(reply.get("dt_s", 0.0)), stem

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


_WORKER: Optional[QnnStemWorkerClient] = None
_WORKER_LOCK = threading.Lock()


def get_qnn_stem_worker(
    *,
    context_bin: Optional[str] = None,
    onnx_path: Optional[str] = None,
) -> QnnStemWorkerClient:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = QnnStemWorkerClient(context_bin=context_bin, onnx_path=onnx_path)
        return _WORKER


def try_get_qnn_stem_worker(
    *,
    context_bin: Optional[str] = None,
    onnx_path: Optional[str] = None,
) -> Optional[QnnStemWorkerClient]:
    if not qnn_stem_persistent_enabled():
        return None
    try:
        return get_qnn_stem_worker(context_bin=context_bin, onnx_path=onnx_path)
    except Exception as exc:
        print(f"[QnnStemWorker] unavailable ({exc})")
        return None


def _worker_main(
    context_bin: Optional[str],
    onnx_path: Optional[str],
    conn: Connection,
) -> None:
    backend = None
    mode = "unknown"
    try:
        import os

        from detectors.npu.qnn_stem_backend import setup_stem_qairt_env

        os.environ.update(setup_stem_qairt_env())
        created = try_create_stem_backend(context_bin=context_bin, onnx_path=onnx_path)
        if created is None:
            raise RuntimeError("no QNN stem backend available")
        backend, mode = created
        conn.send({"ok": True, "mode": mode})
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
            nbytes = int(np.prod(shape))
            raw = conn.recv_bytes()
            if len(raw) != nbytes:
                conn.send({"ok": False, "error": f"payload size {len(raw)} != {nbytes}"})
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            t0 = time.perf_counter()
            dt_s, stem = backend.infer_stem(frame)
            dt_s = time.perf_counter() - t0
            flat = np.ascontiguousarray(stem, dtype=np.float32)
            conn.send({"ok": True, "dt_s": dt_s, "shape": list(flat.shape)})
            conn.send_bytes(flat.tobytes())
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
