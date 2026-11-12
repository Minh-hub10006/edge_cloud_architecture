"""Unix-socket client for native snpe-daemon (persistent conn + /dev/shm I/O)."""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

_MAGIC = 0x45504E53  # SNPE little-endian
_VERSION = 1
_CMD_PING = 1
_CMD_INFER = 2
_CMD_SHUTDOWN = 3
_CMD_INFER_SHM = 4


def _use_shm_infer() -> bool:
    return os.environ.get("SNPE_DAEMON_SHM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class SnpeDaemonClient:
    def __init__(
        self,
        socket_path: str,
        *,
        shm_dir: Optional[str] = None,
        input_bytes: int = 0,
        timeout_s: float = 30.0,
    ):
        self.socket_path = socket_path
        self.shm_dir = shm_dir or ""
        self.input_bytes = int(input_bytes)
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._in_mmap: Optional[np.memmap] = None
        self._in_f32: Optional[np.ndarray] = None
        self._out_maps: Dict[int, Tuple[int, np.memmap]] = {}
        self._shm_enabled = bool(self.shm_dir) and _use_shm_infer() and self.input_bytes > 0
        if self._shm_enabled:
            self._setup_shm()

    def _setup_shm(self) -> None:
        os.makedirs(self.shm_dir, exist_ok=True)
        in_path = os.path.join(self.shm_dir, "in.bin")
        if not os.path.exists(in_path) or os.path.getsize(in_path) != self.input_bytes:
            with open(in_path, "wb") as f:
                f.truncate(self.input_bytes)
        self._in_mmap = np.memmap(
            in_path, dtype=np.uint8, mode="r+", shape=(self.input_bytes,)
        )
        n_floats = self.input_bytes // 4
        self._in_f32 = np.frombuffer(self._in_mmap, dtype=np.float32, count=n_floats)

    def _out_view(self, index: int, n_floats: int) -> np.ndarray:
        path = os.path.join(self.shm_dir, f"out_{index}.bin")
        cached = self._out_maps.get(index)
        if cached is None or cached[0] < n_floats:
            need_bytes = n_floats * 4
            if not os.path.exists(path) or os.path.getsize(path) < need_bytes:
                with open(path, "wb") as f:
                    f.truncate(need_bytes)
            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(n_floats,))
            self._out_maps[index] = (n_floats, mm)
            cached = self._out_maps[index]
        _, mm = cached
        return mm[:n_floats]

    def _write_input_shm(self, arr: np.ndarray) -> None:
        assert self._in_f32 is not None
        flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        if flat.size != self._in_f32.size:
            raise ValueError("input tensor size mismatch for shm")
        self._in_f32[:] = flat

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        sock.settimeout(self.timeout_s)
        sock.connect(self.socket_path)
        return sock

    def _ensure_sock(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        self._sock = self._connect()
        return self._sock

    def _drop_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @staticmethod
    def _read_exact(sock: socket.socket, n: int) -> bytes:
        chunks: list[bytes] = []
        got = 0
        while got < n:
            part = sock.recv(n - got)
            if not part:
                raise ConnectionError("snpe-daemon closed connection")
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def _read_response(self, sock: socket.socket) -> Tuple[float, Dict[str, np.ndarray]]:
        status = self._read_exact(sock, 1)[0]
        if status != 0:
            msg_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            msg = self._read_exact(sock, msg_len).decode("utf-8", errors="replace")
            raise RuntimeError(msg or "snpe-daemon infer failed")

        infer_us = struct.unpack("!I", self._read_exact(sock, 4))[0]
        num_outputs = struct.unpack("!H", self._read_exact(sock, 2))[0]
        outputs: Dict[str, np.ndarray] = {}
        for i in range(num_outputs):
            name_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            name = self._read_exact(sock, name_len).decode("utf-8", errors="replace")
            data_len = struct.unpack("!I", self._read_exact(sock, 4))[0]
            n_floats = data_len // 4
            if self._shm_enabled and self.shm_dir:
                outputs[name] = self._out_view(i, n_floats)
            else:
                raw = self._read_exact(sock, data_len)
                outputs[name] = np.frombuffer(raw, dtype=np.float32).copy()

        dt_s = infer_us / 1_000_000.0 if infer_us else 0.0
        return dt_s, outputs

    def ping(self) -> bool:
        with self._lock:
            try:
                sock = self._ensure_sock()
                sock.sendall(struct.pack("<IBB", _MAGIC, _VERSION, _CMD_PING))
                self._read_response(sock)
                return True
            except OSError:
                self._drop_sock()
                return False

    def infer_raw(self, input_tensor: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        use_shm = self._shm_enabled and self._in_f32 is not None
        if use_shm:
            arr = input_tensor
        else:
            arr = np.ascontiguousarray(input_tensor, dtype=np.float32)
        payload_len = arr.nbytes

        with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(2):
                t0 = time.perf_counter()
                try:
                    sock = self._ensure_sock()
                    if use_shm and payload_len == self.input_bytes:
                        self._write_input_shm(arr)
                        header = struct.pack(
                            "<IBBI", _MAGIC, _VERSION, _CMD_INFER_SHM, payload_len
                        )
                    else:
                        payload = np.ascontiguousarray(input_tensor, dtype=np.float32).tobytes()
                        header = struct.pack(
                            "<IBBI", _MAGIC, _VERSION, _CMD_INFER, len(payload)
                        )
                        sock.sendall(header)
                        sock.sendall(payload)
                        dt_s, outputs = self._read_response(sock)
                        if not dt_s:
                            dt_s = time.perf_counter() - t0
                        return dt_s, outputs

                    sock.sendall(header)
                    dt_s, outputs = self._read_response(sock)
                    if not dt_s:
                        dt_s = time.perf_counter() - t0
                    return dt_s, outputs
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    last_err = e
                    self._drop_sock()
                    if attempt == 0:
                        continue
                    raise
            raise last_err or RuntimeError("snpe-daemon infer failed")

    def shutdown(self) -> None:
        with self._lock:
            try:
                if self._sock is not None:
                    self._sock.sendall(struct.pack("<IBB", _MAGIC, _VERSION, _CMD_SHUTDOWN))
            except OSError:
                pass
            self._drop_sock()
        if self._in_mmap is not None:
            try:
                self._in_mmap.flush()
            except Exception:
                pass

    def close(self) -> None:
        self.shutdown()
