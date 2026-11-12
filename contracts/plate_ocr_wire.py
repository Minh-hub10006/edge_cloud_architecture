"""Edge -> Cloud plate OCR call: one gRPC unary method carrying a JSON header + the JPEG crop.

shortcut: raw-bytes gRPC method instead of a protobuf message, so no protoc regeneration of the shared contract is
needed on every host; move it into edge_video_platform.proto when the contract is next regenerated.
"""
from __future__ import annotations

import json
import struct

METHOD = "/edge_video_platform.v1.PlateOcr/Read"
MAX_JPEG = 2 * 1024 * 1024


def pack(header: dict, jpeg: bytes) -> bytes:
    head = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(head)) + head + jpeg


def unpack(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 4:
        raise ValueError("short plate request")
    (n,) = struct.unpack(">I", data[:4])
    if n > 64 * 1024 or 4 + n > len(data) or len(data) - 4 - n > MAX_JPEG:
        raise ValueError("bad plate request framing")
    header = json.loads(data[4:4 + n])
    if not isinstance(header, dict):
        raise ValueError("plate header must be a JSON object")
    return header, data[4 + n:]


class PlateOcrClient:
    """Async client; read() -> the Cloud's JSON reply {text, calls, ocr_ms, wait_ms, ...}."""

    def __init__(self, target: str, *, timeout_s: float = 5.0) -> None:
        import grpc
        self.timeout_s = timeout_s
        self.channel = grpc.aio.insecure_channel(target, options=(
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024)))
        self._read = self.channel.unary_unary(METHOD)

    async def read(self, header: dict, jpeg: bytes) -> dict:
        return json.loads(await self._read(pack(header, jpeg), timeout=self.timeout_s))

    async def close(self) -> None:
        await self.channel.close()
