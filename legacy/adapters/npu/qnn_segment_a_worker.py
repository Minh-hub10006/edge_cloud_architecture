"""Persistent QNN Segment A worker — keeps ORT QNN or qnn-net-run cache warm."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from multiprocessing.connection import Connection
from typing import Dict, Optional, Tuple

import numpy as np

from detectors.npu.qnn_segment_a_backend import qnn_persistent_enabled, try_create_qnn_backend


class QnnSegmentAWorkerClient:
    def __init__(self, *, onnx_path: Optional[str] = None, model_so: Optional[str] = None):
        self.onnx_path = onnx_path
        self.model_so = model_so
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
            args=(self.onnx_path, self.model_so, child_conn),
            daemon=True,
            name="qnn-segment-a-worker",
        )
        proc.start()
        child_conn.close()
        msg = parent_conn.recv()
        if not isinstance(msg, dict) or not msg.get("ok"):
            raise RuntimeError(msg.get("error", "QNN worker startup failed"))
        self.mode = str(msg.get("mode", "qnn_worker"))
        self._conn = parent_conn
        self._proc = proc

    def infer_tensors(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        if self._conn is None:
            raise RuntimeError("QNN worker not running")
        arr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        with self._lock:
            if self._proc is not None and not self._proc.is_alive():
                raise RuntimeError("QNN worker process exited")
            self._conn.send({"op": "infer", "shape": list(arr.shape), "dtype": "uint8"})
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


_WORKER: Optional[QnnSegmentAWorkerClient] = None
_WORKER_LOCK = threading.Lock()


def get_qnn_worker(
    *,
    onnx_path: Optional[str] = None,
    model_so: Optional[str] = None,
) -> QnnSegmentAWorkerClient:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = QnnSegmentAWorkerClient(onnx_path=onnx_path, model_so=model_so)
        return _WORKER


def try_get_qnn_worker(
    *,
    onnx_path: Optional[str] = None,
    model_so: Optional[str] = None,
) -> Optional[QnnSegmentAWorkerClient]:
    if not qnn_persistent_enabled():
        return None
    try:
        return get_qnn_worker(onnx_path=onnx_path, model_so=model_so)
    except Exception as exc:
        print(f"[QnnSegmentAWorker] unavailable ({exc})")
        return None


def _worker_main(
    onnx_path: Optional[str],
    model_so: Optional[str],
    conn: Connection,
) -> None:
    backend = None
    mode = "unknown"
    try:
        created = try_create_qnn_backend(onnx_path=onnx_path, model_so=model_so)
        if created is None:
            raise RuntimeError("no QNN Segment A backend available")
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
            dt_s, tensors = backend.infer_tensors(frame)
            dt_s = time.perf_counter() - t0
            meta = []
            for name, tensor in tensors.items():
                flat = np.ascontiguousarray(tensor, dtype=np.float32)
                meta.append({"name": name, "shape": list(flat.shape)})
            conn.send({"ok": True, "dt_s": dt_s, "outputs": meta})
            for item in meta:
                conn.send_bytes(tensors[item["name"]].astype(np.float32, copy=False).tobytes())
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
