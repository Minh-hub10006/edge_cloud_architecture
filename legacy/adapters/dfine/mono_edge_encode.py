"""Edge mono path: JPEG-compress frame for full D-FINE on cloud (high FPS LAN)."""

from __future__ import annotations

import os
import time
from typing import Tuple

import cv2
import numpy as np


def mono_jpeg_quality() -> int:
    return int(os.environ.get("MONO_JPEG_QUALITY", "80"))


def encode_frame_jpeg(frame: np.ndarray, *, quality: int | None = None) -> Tuple[float, bytes]:
    """BGR/RGB uint8 HWC → JPEG bytes. Returns (encode_s, payload)."""
    t0 = time.time()
    q = mono_jpeg_quality() if quality is None else int(quality)
    q = max(40, min(95, q))
    img = frame
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode JPEG failed")
    return time.time() - t0, buf.tobytes()
