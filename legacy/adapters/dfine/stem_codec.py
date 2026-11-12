"""Stem cut tensor codec — edge NPU stem → wire → cloud stem decoder."""

from __future__ import annotations

import gzip
import os
from typing import Tuple

import numpy as np

# Edge stem output @ 640×640 input (from extracted stem cut)
STEM_SHAPE = (1, 32, 160, 160)
STEM_ELEM_COUNT = int(np.prod(STEM_SHAPE))  # 819_200
STEM_INT8_HEADER_BYTES = 4  # single float32 scale

# gRPC shape markers (packet.shape[0]):
CUT_STEM_INT8_SHAPE_MARKER = -7.0
CUT_STEM_INT4GZ_SHAPE_MARKER = -8.0


def stem_uplink_mode() -> str:
    """int8 (~800 KB) | int4gz (~65 KB gzip) — default int4gz for cut_stem LAN path."""
    raw = os.environ.get("STEM_UPLINK", "int4gz").strip().lower()
    if raw in ("int4", "int4gz", "4bit", "quant4"):
        return "int4gz"
    return "int8"


def _quantize_symmetric(flat: np.ndarray, n_bits: int) -> tuple[np.ndarray, float]:
    qmax = (1 << (n_bits - 1)) - 1
    amax = float(max(np.max(np.abs(flat)), 1e-6))
    scale = amax / qmax
    q = np.clip(np.round(flat / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale


def _dequantize_symmetric(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def _pack_int4_nibbles(q: np.ndarray) -> bytes:
    x = np.asarray(q, dtype=np.int8).ravel().clip(-8, 7)
    if x.size % 2:
        x = np.concatenate([x, np.zeros(1, dtype=np.int8)])
    x_u = x.astype(np.uint8) & 0x0F
    packed = (x_u[0::2] | (x_u[1::2] << 4)).astype(np.uint8)
    return packed.tobytes()


def _unpack_int4_nibbles(packed: bytes, count: int) -> np.ndarray:
    arr = np.frombuffer(packed, dtype=np.uint8)
    lo = arr & 0x0F
    hi = (arr >> 4) & 0x0F
    out = np.empty(lo.size + hi.size, dtype=np.uint8)
    out[0::2] = lo
    out[1::2] = hi
    out = out[:count]
    signed = np.where(out >= 8, out.astype(np.int16) - 16, out.astype(np.int16))
    return signed.astype(np.int8)


def pack_stem_int8(stem: np.ndarray) -> bytes:
    """Symmetric INT8 pack: 4-byte scale header + int8 body (~800 KB)."""
    flat = np.asarray(stem, dtype=np.float32).ravel()
    if flat.size < STEM_ELEM_COUNT:
        raise ValueError(f"expected {STEM_ELEM_COUNT} floats, got {flat.size}")
    flat = flat[:STEM_ELEM_COUNT]
    q, scale = _quantize_symmetric(flat, 8)
    header = np.asarray([scale], dtype=np.float32).tobytes()
    return header + q.tobytes()


def unpack_stem_int8(payload: bytes) -> np.ndarray:
    """Decode INT8 stem wire bytes → float32 (1,32,160,160)."""
    min_len = STEM_INT8_HEADER_BYTES + STEM_ELEM_COUNT
    if len(payload) < min_len:
        raise ValueError(f"cut_stem_int8 payload too small: {len(payload)}")
    scale = float(np.frombuffer(payload[:STEM_INT8_HEADER_BYTES], dtype=np.float32)[0])
    q = np.frombuffer(payload[STEM_INT8_HEADER_BYTES:], dtype=np.int8)
    if q.size < STEM_ELEM_COUNT:
        raise ValueError(f"expected {STEM_ELEM_COUNT} int8 values, got {q.size}")
    flat = _dequantize_symmetric(q[:STEM_ELEM_COUNT], scale)
    return flat.reshape(STEM_SHAPE)


def pack_stem_int4gz(stem: np.ndarray) -> bytes:
    """INT4 nibble pack + gzip: 4-byte scale + gz body (~65 KB typical)."""
    flat = np.asarray(stem, dtype=np.float32).ravel()
    if flat.size < STEM_ELEM_COUNT:
        raise ValueError(f"expected {STEM_ELEM_COUNT} floats, got {flat.size}")
    flat = flat[:STEM_ELEM_COUNT]
    q, scale = _quantize_symmetric(flat, 4)
    packed = _pack_int4_nibbles(q)
    body = gzip.compress(packed, compresslevel=1)
    header = np.asarray([scale], dtype=np.float32).tobytes()
    return header + body


def unpack_stem_int4gz(payload: bytes) -> np.ndarray:
    """Decode INT4+gzip stem wire bytes → float32 (1,32,160,160)."""
    if len(payload) < STEM_INT8_HEADER_BYTES + 16:
        raise ValueError(f"cut_stem_int4gz payload too small: {len(payload)}")
    scale = float(np.frombuffer(payload[:STEM_INT8_HEADER_BYTES], dtype=np.float32)[0])
    packed = gzip.decompress(payload[STEM_INT8_HEADER_BYTES:])
    q = _unpack_int4_nibbles(packed, STEM_ELEM_COUNT)
    flat = _dequantize_symmetric(q, scale)
    return flat.reshape(STEM_SHAPE)


def pack_stem(stem: np.ndarray) -> tuple[bytes, str]:
    """Pack stem for uplink; returns (payload, layout_name)."""
    if stem_uplink_mode() == "int4gz":
        return pack_stem_int4gz(stem), "cut_stem_int4gz"
    return pack_stem_int8(stem), "cut_stem_int8"


def unpack_stem(payload: bytes, *, layout: str) -> np.ndarray:
    if layout in ("cut_stem_int4gz", "int4gz"):
        return unpack_stem_int4gz(payload)
    return unpack_stem_int8(payload)


def mock_stem() -> Tuple[np.ndarray, np.ndarray]:
    """Random stem for comm tests when NPU/ONNX unavailable."""
    rng = np.random.default_rng(0)
    stem = rng.standard_normal(STEM_SHAPE, dtype=np.float32)
    return stem, pack_stem_int8(stem)
