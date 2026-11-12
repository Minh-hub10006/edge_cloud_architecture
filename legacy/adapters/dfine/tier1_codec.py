"""
Tier-1 tensor codec (Segment A output → wire format).

Aligned with Parallelism_pipeline_AI_detection/scripts/_server_pipeline_server.py
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

# Segment A outputs @ 640×640 input
P3_SHAPE = (1, 384, 80, 80)
P4_SHAPE = (1, 768, 40, 40)
P5_SHAPE = (1, 1536, 20, 20)

P3_N = 384 * 80 * 80   # 2_457_600
P4_N = 768 * 40 * 40   # 1_228_800
P5_N = 1536 * 20 * 20  # 614_400

TIER1_FLOAT_COUNT = P3_N + P4_N + P5_N  # 4_300_800
TIER1_BYTES_FP32 = TIER1_FLOAT_COUNT * 4  # 17_203_200
TIER1_BYTES_INT8 = TIER1_FLOAT_COUNT  # 4_300_800

# gRPC shape marker: tier1 FP32 uses flat_n ≈ TIER1_FLOAT_COUNT; INT8 uses -3
TIER1_INT8_SHAPE_MARKER = -3.0
TIER1_INT8_HEADER_BYTES = 12  # 3 × float32 scales (P3, P4, P5)
TIER1_UINT8_NATIVE_SHAPE_MARKER = -4.0
TIER1_UINT8_NATIVE_HEADER_BYTES = 12  # legacy: 3 float32 scales only

PROJ256_INT8_SHAPE_MARKER = -5.0
PROJ256_INT8_HEADER_BYTES = 12  # 3 x float32 scales (Q3, Q4, Q5)

# Full D-FINE Cloud (mono) sends JPEG bytes. Keep historical split markers
# immutable: -4 is native tier1 and -5 is the projection cut.
MONO_JPEG_SHAPE_MARKER = -6.0

Q3_SHAPE = (1, 256, 80, 80)
Q4_SHAPE = (1, 256, 40, 40)
Q5_SHAPE = (1, 256, 20, 20)
Q3_N = 256 * 80 * 80
Q4_N = 256 * 40 * 40
Q5_N = 256 * 20 * 20
PROJ256_FLOAT_COUNT = Q3_N + Q4_N + Q5_N  # 2_150_400
PROJ256_BYTES_INT8 = PROJ256_FLOAT_COUNT

PROJ256_GLOBAL_AFFINE = {
    "Q3": (1.0497194608868108, -23.623113150494227),
    "Q4": (1.08556943246628, -30.711229245082134),
    "Q5": (1.098739478549779, -9.387711121906559),
}

# Quantization metadata embedded in segment_a_board.serialized.bin. QNN exposes
# the tensors as padded NHWC U8; the split wire format carries only the useful
# channels in NCHW order, followed by Segment B on the cloud.
TIER1_UINT8_NATIVE_SCALES = (0.033842, 0.087245, 0.125106)
# One-time debug dump of the dequantized P3/P4/P5 stats on the cloud, so we can
# compare against a reference forward pass and correct scale/zero-point. Off by
# default => rollback-safe.
TIER1_DEBUG_STATS = os.environ.get("TIER1_DEBUG_STATS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Dequantization zero points (offsets):  real = (q - zero_point) * scale.
# ZERO by default => behaviour identical to the legacy scale-only codec, so this
# is rollback-safe. Override at runtime via env when the QNN export baked a
# non-zero offset (e.g. a constant P4=17 arises from reading at the wrong offset).
TIER1_UINT8_NATIVE_ZERO_POINTS = (0.0, 0.0, 0.0)


def tier1_native_zero_points() -> tuple[float, float, float]:
    raw = os.environ.get("TIER1_UINT8_NATIVE_ZERO_POINTS", "").strip()
    if not raw:
        return TIER1_UINT8_NATIVE_ZERO_POINTS
    try:
        parts = [float(x) for x in raw.replace(",", " ").split() if x.strip()]
    except ValueError:
        return TIER1_UINT8_NATIVE_ZERO_POINTS
    if len(parts) < 3:
        return TIER1_UINT8_NATIVE_ZERO_POINTS
    return (parts[0], parts[1], parts[2])


def tier1_native_scales() -> tuple[float, float, float]:
    raw = os.environ.get("TIER1_UINT8_NATIVE_SCALES", "").strip()
    if not raw:
        return TIER1_UINT8_NATIVE_SCALES
    try:
        parts = [float(x) for x in raw.replace(",", " ").split() if x.strip()]
    except ValueError:
        return TIER1_UINT8_NATIVE_SCALES
    if len(parts) < 3:
        return TIER1_UINT8_NATIVE_SCALES
    return (parts[0], parts[1], parts[2])


TIER1_UINT8_NATIVE_SPECS = (
    ((1, 80, 80, 512), 384),
    ((1, 40, 40, 1024), 768),
    ((1, 20, 20, 2048), 1536),
)

INPUT_SHAPE = (1, 3, 640, 640)


def validate_tier1_tensors(tensors: Dict[str, np.ndarray], *, source: str) -> None:
    """Reject unusable Segment-A features before they reach cloud B+C.

    A bad QNN output previously looked valid because its shapes were correct,
    while P3/P5 were all zero and P4 was a constant 17.  Sending such tensors
    produces plausible-looking but invalid boxes.  This guard is intentionally
    cheap and used only for the first few frames of a backend instance.
    """
    expected = {"P3": P3_SHAPE, "P4": P4_SHAPE, "P5": P5_SHAPE}
    for name, shape in expected.items():
        if name not in tensors:
            raise RuntimeError(f"{source}: missing {name}")
        arr = np.asarray(tensors[name])
        if tuple(arr.shape) != shape:
            raise RuntimeError(f"{source}: {name} shape {tuple(arr.shape)} != {shape}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{source}: {name} contains NaN/Inf")
        # A genuine feature map can be sparse but cannot have exactly one value
        # across all elements.  np.ptp avoids an expensive float64 std pass.
        if float(np.ptp(arr)) <= 1e-12:
            value = float(arr.reshape(-1)[0]) if arr.size else float("nan")
            raise RuntimeError(
                f"{source}: {name} is constant ({value}); refusing invalid split payload"
            )


def validate_proj256_tensors(tensors: Dict[str, np.ndarray], *, source: str) -> None:
    """Reject unusable projection-cut features before they reach cloud B+C."""
    expected = {"Q3": Q3_SHAPE, "Q4": Q4_SHAPE, "Q5": Q5_SHAPE}
    for name, shape in expected.items():
        if name not in tensors:
            raise RuntimeError(f"{source}: missing {name}")
        arr = np.asarray(tensors[name])
        if tuple(arr.shape) != shape:
            raise RuntimeError(f"{source}: {name} shape {tuple(arr.shape)} != {shape}")
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{source}: {name} contains NaN/Inf")
        if float(np.ptp(arr)) <= 1e-12:
            value = float(arr.reshape(-1)[0]) if arr.size else float("nan")
            raise RuntimeError(
                f"{source}: {name} is constant ({value}); refusing invalid proj256 payload"
            )


def tier1_uplink_int8() -> bool:
    return os.environ.get("TIER1_UPLINK", "fp32").strip().lower() in (
        "int8",
        "tier1_int8",
        "quant",
    )


def tier1_native_uint8_enabled() -> bool:
    """Opt in to native QNN U8 output; keep the INT8 codec as rollback."""
    raw = os.environ.get("DFINE_QNN_NATIVE_OUTPUT", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def flatten_tier1(P3: np.ndarray, P4: np.ndarray, P5: np.ndarray) -> np.ndarray:
    """Concatenate P3, P4, P5 to 1D float32 (layout: tier1_flat)."""
    return np.concatenate(
        [P3.astype(np.float32).ravel(), P4.astype(np.float32).ravel(), P5.astype(np.float32).ravel()]
    )


def unflatten_tier1(flat: np.ndarray) -> Dict[str, np.ndarray]:
    """Decode tier1_flat buffer → named tensors for Segment B."""
    flat = np.asarray(flat, dtype=np.float32).ravel()
    if flat.size < TIER1_FLOAT_COUNT:
        raise ValueError(f"expected {TIER1_FLOAT_COUNT} floats, got {flat.size}")
    P3 = flat[:P3_N].reshape(P3_SHAPE)
    P4 = flat[P3_N : P3_N + P4_N].reshape(P4_SHAPE)
    P5 = flat[P3_N + P4_N :].reshape(P5_SHAPE)
    return {"P3": P3, "P4": P4, "P5": P5}


def _quantize_tensor(t: np.ndarray) -> tuple[np.ndarray, float]:
    f = t.astype(np.float32).ravel()
    scale = float(max(np.max(np.abs(f)), 1e-6) / 127.0)
    q = np.clip(np.round(f / scale), -127, 127).astype(np.int8)
    return q, scale


def pack_tier1_int8(P3: np.ndarray, P4: np.ndarray, P5: np.ndarray) -> bytes:
    """Symmetric per-tensor INT8 pack: 12-byte scales header + int8 body (~4.3 MB)."""
    parts: list[np.ndarray] = []
    scales: list[float] = []
    for t in (P3, P4, P5):
        q, scale = _quantize_tensor(t)
        parts.append(q)
        scales.append(scale)
    header = np.asarray(scales, dtype=np.float32).tobytes()
    body = np.concatenate(parts).tobytes()
    return header + body


def unpack_tier1_int8(payload: bytes) -> Dict[str, np.ndarray]:
    """Decode INT8 tier1 wire bytes → float32 P3/P4/P5 for Segment B."""
    if len(payload) < TIER1_INT8_HEADER_BYTES + TIER1_BYTES_INT8:
        raise ValueError(f"tier1_int8 payload too small: {len(payload)}")
    scales = np.frombuffer(payload[:TIER1_INT8_HEADER_BYTES], dtype=np.float32)
    flat_q = np.frombuffer(payload[TIER1_INT8_HEADER_BYTES:], dtype=np.int8)
    if flat_q.size < TIER1_FLOAT_COUNT:
        raise ValueError(f"expected {TIER1_FLOAT_COUNT} int8 values, got {flat_q.size}")
    flat_f = flat_q[:TIER1_FLOAT_COUNT].astype(np.float32)
    P3 = (flat_f[:P3_N] * scales[0]).reshape(P3_SHAPE)
    P4 = (flat_f[P3_N : P3_N + P4_N] * scales[1]).reshape(P4_SHAPE)
    P5 = (flat_f[P3_N + P4_N :] * scales[2]).reshape(P5_SHAPE)
    return {"P3": P3, "P4": P4, "P5": P5}


def proj256_affine_coeffs() -> dict[str, tuple[float, float]]:
    """Return projection-cut affine coeffs, with optional env override.

    Override format:
      DFINE_PROJ256_AFFINE="s3,b3,s4,b4,s5,b5"
    """
    raw = os.environ.get("DFINE_PROJ256_AFFINE", "").strip()
    if not raw:
        return dict(PROJ256_GLOBAL_AFFINE)
    try:
        vals = [float(x) for x in raw.replace(",", " ").split() if x.strip()]
    except ValueError:
        return dict(PROJ256_GLOBAL_AFFINE)
    if len(vals) != 6:
        return dict(PROJ256_GLOBAL_AFFINE)
    return {
        "Q3": (vals[0], vals[1]),
        "Q4": (vals[2], vals[3]),
        "Q5": (vals[4], vals[5]),
    }


def apply_proj256_affine(tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    coeffs = proj256_affine_coeffs()
    corrected: Dict[str, np.ndarray] = {}
    for key in ("Q3", "Q4", "Q5"):
        scale, bias = coeffs[key]
        value = np.asarray(tensors[key], dtype=np.float32) * scale + bias
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite corrected proj256 tensor {key}")
        corrected[key] = value.astype(np.float32, copy=False)
    return corrected


def pack_proj256_int8(
    Q3: np.ndarray,
    Q4: np.ndarray,
    Q5: np.ndarray,
    *,
    apply_affine: bool = True,
) -> bytes:
    """Projection-cut symmetric INT8 pack: 12-byte scales + 2.15 MB body."""
    tensors = {"Q3": Q3, "Q4": Q4, "Q5": Q5}
    if apply_affine:
        tensors = apply_proj256_affine(tensors)
    parts: list[np.ndarray] = []
    scales: list[float] = []
    for key in ("Q3", "Q4", "Q5"):
        q, scale = _quantize_tensor(tensors[key])
        parts.append(q)
        scales.append(scale)
    payload = np.asarray(scales, dtype=np.float32).tobytes() + np.concatenate(parts).tobytes()
    expected = PROJ256_INT8_HEADER_BYTES + PROJ256_BYTES_INT8
    if len(payload) != expected:
        raise RuntimeError(f"proj256_int8 payload size {len(payload)} != {expected}")
    return payload


def unpack_proj256_int8(payload: bytes) -> Dict[str, np.ndarray]:
    """Decode projection-cut INT8 wire bytes into float32 Q3/Q4/Q5."""
    expected = PROJ256_INT8_HEADER_BYTES + PROJ256_BYTES_INT8
    if len(payload) < expected:
        raise ValueError(f"proj256_int8 payload too small: {len(payload)}")
    scales = np.frombuffer(payload[:PROJ256_INT8_HEADER_BYTES], dtype=np.float32)
    flat_q = np.frombuffer(payload[PROJ256_INT8_HEADER_BYTES:], dtype=np.int8)
    if flat_q.size < PROJ256_FLOAT_COUNT:
        raise ValueError(f"expected {PROJ256_FLOAT_COUNT} int8 values, got {flat_q.size}")
    flat_f = flat_q[:PROJ256_FLOAT_COUNT].astype(np.float32)
    Q3 = (flat_f[:Q3_N] * scales[0]).reshape(Q3_SHAPE)
    Q4 = (flat_f[Q3_N : Q3_N + Q4_N] * scales[1]).reshape(Q4_SHAPE)
    Q5 = (flat_f[Q3_N + Q4_N :] * scales[2]).reshape(Q5_SHAPE)
    return {"Q3": Q3, "Q4": Q4, "Q5": Q5}


def pack_tier1_uint8_native(
    P3_nhwc: np.ndarray,
    P4_nhwc: np.ndarray,
    P5_nhwc: np.ndarray,
) -> bytes:
    """Pack QNN-native padded NHWC U8 outputs without requantization."""
    bodies: list[bytes] = []
    for name, tensor, (native_shape, useful_channels) in zip(
        ("P3", "P4", "P5"),
        (P3_nhwc, P4_nhwc, P5_nhwc),
        TIER1_UINT8_NATIVE_SPECS,
    ):
        arr = np.asarray(tensor)
        if arr.dtype != np.uint8:
            raise TypeError(f"{name} native dtype {arr.dtype} != uint8")
        if tuple(arr.shape) != native_shape:
            raise ValueError(f"{name} native shape {arr.shape} != {native_shape}")
        nchw = np.transpose(arr, (0, 3, 1, 2))[:, :useful_channels, :, :]
        bodies.append(np.ascontiguousarray(nchw).tobytes())
    header = np.asarray(tier1_native_scales(), dtype=np.float32).tobytes()
    payload = header + b"".join(bodies)
    expected = TIER1_UINT8_NATIVE_HEADER_BYTES + TIER1_BYTES_INT8
    if len(payload) != expected:
        raise RuntimeError(f"native uint8 payload size {len(payload)} != {expected}")
    return payload


def unpack_tier1_uint8_native(payload: bytes) -> Dict[str, np.ndarray]:
    """Decode native QNN U8 payload into float32 NCHW tensors for Segment B."""
    expected = TIER1_UINT8_NATIVE_HEADER_BYTES + TIER1_BYTES_INT8
    if len(payload) < expected:
        raise ValueError(f"tier1_uint8_native payload too small: {len(payload)}")
    scales = np.frombuffer(
        payload[:TIER1_UINT8_NATIVE_HEADER_BYTES], dtype=np.float32
    )
    flat_q = np.frombuffer(
        payload[TIER1_UINT8_NATIVE_HEADER_BYTES:], dtype=np.uint8
    )
    if flat_q.size < TIER1_FLOAT_COUNT:
        raise ValueError(
            f"expected {TIER1_FLOAT_COUNT} uint8 values, got {flat_q.size}"
        )
    flat_f = flat_q[:TIER1_FLOAT_COUNT].astype(np.float32)
    zp = np.asarray(tier1_native_zero_points(), dtype=np.float32)
    scales = np.asarray(tier1_native_scales(), dtype=np.float32)
    P3 = ((flat_f[:P3_N] - zp[0]) * scales[0]).reshape(P3_SHAPE)
    P4 = ((flat_f[P3_N : P3_N + P4_N] - zp[1]) * scales[1]).reshape(P4_SHAPE)
    P5 = ((flat_f[P3_N + P4_N :] - zp[2]) * scales[2]).reshape(P5_SHAPE)
    _dump_tier1_stats(P3, P4, P5, scales, zp)
    return {"P3": P3, "P4": P4, "P5": P5}


def _dump_tier1_stats(
    P3: np.ndarray,
    P4: np.ndarray,
    P5: np.ndarray,
    scales: np.ndarray,
    zp: np.ndarray,
) -> None:
    """Print one-time dequantized feature stats (off unless TIER1_DEBUG_STATS=1)."""
    if not TIER1_DEBUG_STATS:
        return
    if getattr(_dump_tier1_stats, "_printed", False):
        return
    _dump_tier1_stats._printed = True
    for name, t in (("P3", P3), ("P4", P4), ("P5", P5)):
        arr = np.asarray(t, dtype=np.float32)
        print(
            f"[TIER1 DEBUG] {name} shape={arr.shape} min={float(arr.min()):.4f} "
            f"max={float(arr.max()):.4f} mean={float(arr.mean()):.4f} "
            f"absmax={float(np.max(np.abs(arr))):.4f}",
            flush=True,
        )
    print(
        f"[TIER1 DEBUG] scale={[float(s) for s in scales]} zp={[float(z) for z in zp]}",
        flush=True,
    )


def mock_tier1() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Random tier1 for comm tests when ONNX/QNN not available."""
    rng = np.random.default_rng(0)
    P3 = rng.standard_normal(P3_SHAPE, dtype=np.float32)
    P4 = rng.standard_normal(P4_SHAPE, dtype=np.float32)
    P5 = rng.standard_normal(P5_SHAPE, dtype=np.float32)
    flat = flatten_tier1(P3, P4, P5)
    return flat, {"P3": P3, "P4": P4, "P5": P5}
