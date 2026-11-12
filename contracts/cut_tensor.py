"""Path B wire format (decision 5/11): the Edge head's cut tensor as uint8 (one min/max per tensor) + zstd level 1
(6/11: 2.7x faster than level 3 and no bigger on real tensors, 449 vs 454 KB).

Layout: 38-byte header (magic, head id, min, max, source width/height, C/H/W) + zstd(uint8 NCHW values).
The head id names the Edge half that produced the tensor; the Cloud refuses a tensor from another head.
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = b"CUT1"
_HEADER = struct.Struct("!4s16sffHHHHH")  # magic, head id, lo, hi, width, height, C, H, W


def pack(t: np.ndarray, *, width: int, height: int, head_id: str, level: int = 1) -> bytes:
    """t: float32 [1, C, H, W] (NCHW) -> bytes."""
    import zstandard
    if t.ndim != 4 or t.shape[0] != 1:
        raise ValueError("cut tensor must be [1, C, H, W]")
    head = head_id.encode()
    if not head or len(head) > 16:
        raise ValueError("head_id must be 1..16 ASCII bytes")
    lo, hi = float(t.min()), float(t.max())
    q = np.round((t - lo) * (255 / max(hi - lo, 1e-12))).astype(np.uint8)
    _, c, h, w = t.shape
    return _HEADER.pack(MAGIC, head, lo, hi, width, height, c, h, w) + zstandard.ZstdCompressor(level=level).compress(q.tobytes())


def unpack(data: bytes, *, expected_head_id) -> tuple[np.ndarray, int, int]:
    """bytes -> (float32 [1, C, H, W], source width, source height); ValueError on a bad or foreign tensor."""
    q, lo, hi, width, height = unpack_u8(data, expected_head_id=expected_head_id)
    return q.astype(np.float32) * np.float32((hi - lo) / 255) + np.float32(lo), width, height


def unpack_u8(data: bytes, *, expected_head_id) -> tuple[np.ndarray, float, float, int, int]:
    """bytes -> (uint8 [1, C, H, W], min, max, source width, source height), for a tail that dequantizes itself.
    expected_head_id: one head id or a collection of accepted ones (NPU and GPU Edges cut the same model)."""
    import zstandard
    if len(data) <= _HEADER.size:
        raise ValueError("cut tensor too short")
    magic, head, lo, hi, width, height, c, h, w = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("not a cut tensor")
    sender = head.rstrip(b"\0").decode(errors="replace")  # no backslash inside f-strings: the Cloud env is Python 3.10
    allowed = {expected_head_id} if isinstance(expected_head_id, str) else set(expected_head_id)
    if sender not in allowed:
        raise ValueError(f"cut tensor from head {sender!r}, expected {sorted(allowed)!r}")
    n = c * h * w
    raw = zstandard.ZstdDecompressor().decompress(data[_HEADER.size:], max_output_size=n)
    if len(raw) != n or not width or not height:
        raise ValueError("cut tensor size does not match its header")
    return np.frombuffer(raw, dtype=np.uint8).reshape(1, c, h, w), lo, hi, width, height
