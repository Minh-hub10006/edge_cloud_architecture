"""Unix-socket client for native qnn-daemon (persistent conn + /dev/shm I/O).

Wire protocol is byte-compatible with snpe-daemon (same header layout) but uses
the 'QNN!' magic word. Outputs are float32 NHWC graph tensors (P3/P4/P5).
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

_MAGIC = 0x214E4E51  # 'QNN!' little-endian
_VERSION = 1
_CMD_PING = 1
_CMD_INFER = 2
_CMD_SHUTDOWN = 3
_CMD_INFER_SHM = 4
_CMD_INFER_NATIVE = 5
_CMD_INFER_TIER1_PACKED_SHM = 6
_CMD_INFER_PROJ256_PACKED_SHM = 7


def _use_shm_infer() -> bool:
    return os.environ.get("QNN_DAEMON_SHM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class QnnDaemonClient:
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
        self._shm_enabled = bool(self.shm_dir) and _use_shm_infer() and self.input_bytes > 0
        if self._shm_enabled:
            self._setup_shm()

    def _setup_shm(self) -> None:
        os.makedirs(self.shm_dir, exist_ok=True)
        in_path = os.path.join(self.shm_dir, "in.bin")
        if not os.path.exists(in_path) or os.path.getsize(in_path) != self.input_bytes:
            with open(in_path, "wb") as f:
                f.truncate(self.input_bytes)
        self._in_mmap = np.memmap(in_path, dtype=np.uint8, mode="r+", shape=(self.input_bytes,))
        n_floats = self.input_bytes // 4
        self._in_f32 = np.frombuffer(self._in_mmap, dtype=np.float32, count=n_floats)

    def _out_view(self, index: int, n_floats: int) -> np.ndarray:
        """Read one completed daemon output into process-owned memory.

        The daemon opens each output with O_TRUNC on every inference.  Keeping
        a numpy memmap of that inode across requests creates a race where the
        mapped file temporarily has length zero, which Linux reports as
        SIGBUS.  Metadata is sent only after the write completes, so a normal
        read here is safe and still avoids sending the large tensors through
        the Unix socket.
        """
        path = os.path.join(self.shm_dir, f"out_{index}.bin")
        expected_bytes = n_floats * 4
        actual_bytes = os.path.getsize(path)
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"QNN shm output {index} size {actual_bytes} != {expected_bytes}"
            )
        out = np.fromfile(path, dtype=np.float32, count=n_floats)
        if out.size != n_floats:
            raise RuntimeError(
                f"short QNN shm output {index}: {out.size} != {n_floats}"
            )
        return out

    def _write_input_shm(self, arr: np.ndarray) -> None:
        assert self._in_f32 is not None
        flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        if flat.size != self._in_f32.size:
            raise ValueError("input tensor size mismatch for shm")
        self._in_f32[:] = flat

    def _write_input_bytes_shm(self, payload: bytes) -> None:
        assert self._in_mmap is not None
        if len(payload) != self.input_bytes:
            raise ValueError("native input size mismatch for shm")
        self._in_mmap[:] = np.frombuffer(payload, dtype=np.uint8)

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
                raise ConnectionError("qnn-daemon closed connection")
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def _read_response(
        self, sock: socket.socket, *, outputs_in_shm: bool = False
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        status = self._read_exact(sock, 1)[0]
        if status != 0:
            msg_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            msg = self._read_exact(sock, msg_len).decode("utf-8", errors="replace")
            raise RuntimeError(msg or "qnn-daemon infer failed")

        infer_us = struct.unpack("!I", self._read_exact(sock, 4))[0]
        num_outputs = struct.unpack("!H", self._read_exact(sock, 2))[0]
        outputs: Dict[str, np.ndarray] = {}
        for i in range(num_outputs):
            name_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            name = self._read_exact(sock, name_len).decode("utf-8", errors="replace")
            data_len = struct.unpack("!I", self._read_exact(sock, 4))[0]
            n_floats = data_len // 4
            if outputs_in_shm and self.shm_dir:
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
            except (OSError, ConnectionError, RuntimeError):
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
                        dt_s, outputs = self._read_response(sock, outputs_in_shm=False)
                        if not dt_s:
                            dt_s = time.perf_counter() - t0
                        return dt_s, outputs

                    sock.sendall(header)
                    dt_s, outputs = self._read_response(sock, outputs_in_shm=True)
                    if not dt_s:
                        dt_s = time.perf_counter() - t0
                    return dt_s, outputs
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    last_err = e
                    self._drop_sock()
                    if attempt == 0:
                        continue
                    raise
            raise last_err or RuntimeError("qnn-daemon infer failed")

    def infer_bytes(self, payload: bytes) -> Tuple[float, Dict[str, np.ndarray]]:
        """Send raw input bytes (native uint8) over the socket; read float outputs."""
        with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(2):
                t0 = time.perf_counter()
                try:
                    sock = self._ensure_sock()
                    use_shm = self._shm_enabled and len(payload) == self.input_bytes
                    if use_shm:
                        self._write_input_bytes_shm(payload)
                        sock.sendall(
                            struct.pack(
                                "<IBBI", _MAGIC, _VERSION, _CMD_INFER_SHM, len(payload)
                            )
                        )
                    else:
                        sock.sendall(
                            struct.pack(
                                "<IBBI", _MAGIC, _VERSION, _CMD_INFER, len(payload)
                            )
                        )
                        sock.sendall(payload)
                    dt_s, outputs = self._read_response(
                        sock, outputs_in_shm=use_shm
                    )
                    if not dt_s:
                        dt_s = time.perf_counter() - t0
                    return dt_s, outputs
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    last_err = e
                    self._drop_sock()
                    if attempt == 0:
                        continue
                    raise
            raise last_err or RuntimeError("qnn-daemon infer failed")

    def infer_bytes_native(self, payload: bytes) -> Tuple[float, Dict[str, np.ndarray]]:
        """Send raw input bytes (native uint8); read native quantized uint8/int8 outputs."""
        with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(2):
                t0 = time.perf_counter()
                try:
                    sock = self._ensure_sock()
                    sock.sendall(
                        struct.pack("<IBBI", _MAGIC, _VERSION, _CMD_INFER_NATIVE, len(payload))
                    )
                    sock.sendall(payload)
                    dt_s, outputs = self._read_response_native(sock)
                    if not dt_s:
                        dt_s = time.perf_counter() - t0
                    return dt_s, outputs
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    last_err = e
                    self._drop_sock()
                    if attempt == 0:
                        continue
                    raise
            raise last_err or RuntimeError("qnn-daemon native infer failed")

    def infer_tier1_packed_shm(
        self, payload: bytes, affine: tuple[float, float, float, float, float, float]
    ) -> Tuple[float, bytes]:
        """Run verified W8A16 Segment-A and receive its already-packed Tier-1 INT8 wire payload.

        Command 6 deliberately requires the existing shared-memory input path.
        It never creates Python FP32 P3/P4/P5 tensors, so it is opt-in and
        fails closed if the daemon/client was not configured for shared memory.
        """
        if not self._shm_enabled or len(payload) != self.input_bytes:
            raise RuntimeError("tier1 packed QNN command requires configured shared-memory input")
        if len(affine) != 6 or not np.isfinite(np.asarray(affine, dtype=np.float32)).all():
            raise ValueError("tier1 packed affine must contain six finite floats")
        with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(2):
                t0 = time.perf_counter()
                try:
                    sock = self._ensure_sock()
                    self._write_input_bytes_shm(payload)
                    sock.sendall(struct.pack(
                        "<IBBI6f", _MAGIC, _VERSION, _CMD_INFER_TIER1_PACKED_SHM,
                        len(payload), *affine,
                    ))
                    dt_s, raw = self._read_tier1_packed_response(sock)
                    if not dt_s:
                        dt_s = time.perf_counter() - t0
                    return dt_s, raw
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    last_err = e
                    self._drop_sock()
                    if attempt == 0:
                        continue
                    raise
            raise last_err or RuntimeError("qnn-daemon tier1 packed infer failed")

    def infer_proj256_packed_shm(
        self, payload: bytes, affine: tuple[float, float, float, float, float, float]
    ) -> Tuple[float, bytes]:
        """Run projection-cut Segment-A and receive native Q3/Q4/Q5 INT8 bytes."""
        if not self._shm_enabled or len(payload) != self.input_bytes:
            raise RuntimeError("proj256 packed QNN command requires configured shared-memory input")
        if len(affine) != 6 or not np.isfinite(np.asarray(affine, dtype=np.float32)).all():
            raise ValueError("proj256 packed affine must contain six finite floats")
        with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(2):
                t0 = time.perf_counter()
                try:
                    sock = self._ensure_sock()
                    self._write_input_bytes_shm(payload)
                    sock.sendall(struct.pack(
                        "<IBBI6f", _MAGIC, _VERSION, _CMD_INFER_PROJ256_PACKED_SHM,
                        len(payload), *affine,
                    ))
                    dt_s, raw = self._read_packed_response(sock, "proj256_int8")
                    if not dt_s:
                        dt_s = time.perf_counter() - t0
                    return dt_s, raw
                except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                    last_err = e
                    self._drop_sock()
                    if attempt == 0:
                        continue
                    raise
            raise last_err or RuntimeError("qnn-daemon proj256 packed infer failed")

    def _read_tier1_packed_response(self, sock: socket.socket) -> Tuple[float, bytes]:
        """Read command-6 response as bytes, never interpreting the INT8 body as FP32."""
        return self._read_packed_response(sock, "tier1_int8")

    def _read_packed_response(self, sock: socket.socket, expected_name: str) -> Tuple[float, bytes]:
        """Read a native packed response without interpreting its signed INT8 body."""
        status = self._read_exact(sock, 1)[0]
        if status != 0:
            msg_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            msg = self._read_exact(sock, msg_len).decode("utf-8", errors="replace")
            raise RuntimeError(msg or f"qnn-daemon {expected_name} packed infer failed")
        infer_us = struct.unpack("!I", self._read_exact(sock, 4))[0]
        num_outputs = struct.unpack("!H", self._read_exact(sock, 2))[0]
        if num_outputs != 1:
            raise RuntimeError(f"{expected_name} packed response has {num_outputs} outputs, expected one")
        name_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
        name = self._read_exact(sock, name_len).decode("utf-8", errors="replace")
        data_len = struct.unpack("!I", self._read_exact(sock, 4))[0]
        if name != expected_name:
            raise RuntimeError(f"packed response name {name!r} != {expected_name!r}")
        raw = self._read_exact(sock, data_len)
        return infer_us / 1_000_000.0 if infer_us else 0.0, raw

    def _read_response_native(self, sock: socket.socket) -> Tuple[float, Dict[str, np.ndarray]]:
        status = self._read_exact(sock, 1)[0]
        if status != 0:
            msg_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            msg = self._read_exact(sock, msg_len).decode("utf-8", errors="replace")
            raise RuntimeError(msg or "qnn-daemon native infer failed")

        infer_us = struct.unpack("!I", self._read_exact(sock, 4))[0]
        num_outputs = struct.unpack("!H", self._read_exact(sock, 2))[0]
        outputs: Dict[str, np.ndarray] = {}
        for i in range(num_outputs):
            name_len = struct.unpack("!H", self._read_exact(sock, 2))[0]
            name = self._read_exact(sock, name_len).decode("utf-8", errors="replace")
            data_len = struct.unpack("!I", self._read_exact(sock, 4))[0]
            raw = self._read_exact(sock, data_len)
            outputs[name] = np.frombuffer(raw, dtype=np.uint8).copy()
        dt_s = infer_us / 1_000_000.0 if infer_us else 0.0
        return dt_s, outputs

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
        """Close this connection without sending SHUTDOWN to a shared daemon."""
        with self._lock:
            self._drop_sock()
        if self._in_mmap is not None:
            try:
                self._in_mmap.flush()
            except Exception:
                pass
