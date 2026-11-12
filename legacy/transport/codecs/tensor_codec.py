"""Serialize numpy feature tensors for gRPC FeaturePacket.payload."""

from __future__ import annotations

import gzip
from typing import Tuple

import numpy as np

from comm.types.messages import TensorMeta


def encode_tensor(
    arr: np.ndarray,
    compression: str = "none",
) -> Tuple[bytes, TensorMeta]:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    raw = arr.tobytes()
    if compression == "gzip":
        raw = gzip.compress(raw)
    meta = TensorMeta(
        shape=list(arr.shape),
        dtype="float32",
        layout="C",
        compression=compression,
    )
    return raw, meta


def decode_tensor(payload: bytes, meta: TensorMeta) -> np.ndarray:
    data = payload
    if meta.compression == "gzip":
        data = gzip.decompress(data)
    arr = np.frombuffer(data, dtype=np.dtype(meta.dtype))
    return arr.reshape(tuple(meta.shape))
