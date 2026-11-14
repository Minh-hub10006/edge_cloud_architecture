"""Path B wire format.

CUT1 (decision 5/11): the Edge head's cut tensor as uint8 (one min/max per tensor) + zstd level 1 (~450 KB).
  38-byte header (magic, head id, min, max, source width/height, C/H/W) + zstd(uint8 NCHW values).
CUT2 (14/11, default): `bits` per value (2 or 4) with one min/max per channel, values packed into bytes, + zstd.
  2 bits: ~93 KB on UA-DETRAC, F1 0.839 vs 0.8445 for CUT1 (scripts/compress_survey.py: a global range at 2 bits
  zeroes the weak channels, F1 0.545). 33-byte header (magic, head id, bits, source width/height, C/H/W)
  + C float32 minima + C float32 maxima (little endian) + zstd(packed codes).
The head id names the Edge half that produced the tensor; the Cloud refuses a tensor from another head.
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = b"CUT1"
MAGIC_Q = b"CUT2"
_HEADER = struct.Struct("!4s16sffHHHHH")    # magic, head id, lo, hi, width, height, C, H, W
_HEADER_Q = struct.Struct("!4s16sBHHHHH")   # magic, head id, bits, width, height, C, H, W


def _head_bytes(head_id: str) -> bytes:
    head = head_id.encode()
    if not head or len(head) > 16:
        raise ValueError("head_id must be 1..16 ASCII bytes")
    return head


def _pack_codes(q: np.ndarray, bits: int) -> bytes:
    per = 8 // bits
    flat = q.reshape(-1)
    flat = np.concatenate([flat, np.zeros((-flat.size) % per, np.uint8)]).reshape(-1, per)
    out = np.zeros(flat.shape[0], np.uint8)
    for k in range(per):   # first value in the high bits
        out |= flat[:, k] << np.uint8(bits * (per - 1 - k))
    return out.tobytes()


def _unpack_codes(raw: bytes, bits: int, n: int) -> np.ndarray:
    per, mask = 8 // bits, np.uint8(2 ** bits - 1)
    b = np.frombuffer(raw, np.uint8)
    return np.stack([(b >> np.uint8(bits * (per - 1 - k))) & mask for k in range(per)], axis=1).reshape(-1)[:n]


def pack(t: np.ndarray, *, width: int, height: int, head_id: str, level: int = 1, bits: int = 8) -> bytes:
    """t: float32 [1, C, H, W] (NCHW) -> bytes. bits 8 = CUT1, 2 or 4 = CUT2 (per-channel ranges)."""
    import zstandard
    if t.ndim != 4 or t.shape[0] != 1:
        raise ValueError("cut tensor must be [1, C, H, W]")
    head = _head_bytes(head_id)
    _, c, h, w = t.shape
    if bits == 8:
        lo, hi = float(t.min()), float(t.max())
        q = np.round((t - lo) * (255 / max(hi - lo, 1e-12))).astype(np.uint8)
        return (_HEADER.pack(MAGIC, head, lo, hi, width, height, c, h, w)
                + zstandard.ZstdCompressor(level=level).compress(q.tobytes()))
    if bits not in (2, 4):
        raise ValueError("bits must be 8, 4 or 2")
    lo = t[0].min(axis=(1, 2)).astype("<f4")
    hi = t[0].max(axis=(1, 2)).astype("<f4")
    scale = (2 ** bits - 1) / np.maximum(hi - lo, 1e-12)
    q = np.round((t[0] - lo[:, None, None]) * scale[:, None, None]).astype(np.uint8)
    return (_HEADER_Q.pack(MAGIC_Q, head, bits, width, height, c, h, w) + lo.tobytes() + hi.tobytes()
            + zstandard.ZstdCompressor(level=level).compress(_pack_codes(q, bits)))


def _check_head(head: bytes, expected_head_id) -> None:
    sender = head.rstrip(b"\0").decode(errors="replace")  # no backslash inside f-strings: the Cloud env is Python 3.10
    allowed = {expected_head_id} if isinstance(expected_head_id, str) else set(expected_head_id)
    if sender not in allowed:
        raise ValueError(f"cut tensor from head {sender!r}, expected {sorted(allowed)!r}")


def unpack_codes(data: bytes, *, expected_head_id) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """CUT1 or CUT2 bytes -> (codes uint8 [1, C, H, W], lo float32 [1, C], step float32 [1, C], width, height):
    value = code * step + lo per channel, what the rtdetr_tail_q model computes on the GPU."""
    import zstandard
    if len(data) < 4:
        raise ValueError("cut tensor too short")
    if data[:4] == MAGIC:
        q, lo, hi, width, height = unpack_u8(data, expected_head_id=expected_head_id)
        c = q.shape[1]
        return (q, np.full((1, c), lo, np.float32), np.full((1, c), (hi - lo) / 255, np.float32), width, height)
    if data[:4] != MAGIC_Q or len(data) <= _HEADER_Q.size:
        raise ValueError("not a cut tensor")
    _, head, bits, width, height, c, h, w = _HEADER_Q.unpack_from(data)
    _check_head(head, expected_head_id)
    if bits not in (2, 4) or not width or not height:
        raise ValueError("cut tensor header is invalid")
    off = _HEADER_Q.size
    if len(data) <= off + 8 * c:
        raise ValueError("cut tensor too short")
    lo = np.frombuffer(data, "<f4", c, off).astype(np.float32)
    hi = np.frombuffer(data, "<f4", c, off + 4 * c).astype(np.float32)
    n = c * h * w
    packed = -(-n * bits // 8)
    try:
        raw = zstandard.ZstdDecompressor().decompress(data[off + 8 * c:], max_output_size=packed)
    except zstandard.ZstdError as exc:   # a cut-off packet: a bad tensor, not a crash
        raise ValueError(f"cut tensor does not decompress: {exc}") from None
    if len(raw) != packed:
        raise ValueError("cut tensor size does not match its header")
    q = _unpack_codes(raw, bits, n).reshape(1, c, h, w)
    return q, lo[None], ((hi - lo) / (2 ** bits - 1))[None].astype(np.float32), width, height


def unpack(data: bytes, *, expected_head_id) -> tuple[np.ndarray, int, int]:
    """bytes (CUT1 or CUT2) -> (float32 [1, C, H, W], source width, source height); ValueError on a bad or foreign
    tensor."""
    q, lo, step, width, height = unpack_codes(data, expected_head_id=expected_head_id)
    return q.astype(np.float32) * step[:, :, None, None] + lo[:, :, None, None], width, height


def unpack_u8(data: bytes, *, expected_head_id) -> tuple[np.ndarray, float, float, int, int]:
    """CUT1 bytes -> (uint8 [1, C, H, W], min, max, source width, source height).
    expected_head_id: one head id or a collection of accepted ones (NPU and GPU Edges cut the same model)."""
    import zstandard
    if len(data) <= _HEADER.size:
        raise ValueError("cut tensor too short")
    magic, head, lo, hi, width, height, c, h, w = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("not a cut tensor")
    _check_head(head, expected_head_id)
    n = c * h * w
    try:
        raw = zstandard.ZstdDecompressor().decompress(data[_HEADER.size:], max_output_size=n)
    except zstandard.ZstdError as exc:
        raise ValueError(f"cut tensor does not decompress: {exc}") from None
    if len(raw) != n or not width or not height:
        raise ValueError("cut tensor size does not match its header")
    return np.frombuffer(raw, dtype=np.uint8).reshape(1, c, h, w), lo, hi, width, height
