"""UC3 crowd density (P2, 10/11): DM-Count (MIT, ShanghaiTech B weights) pre- and post-processing, shared by the Cloud
worker (cloud_core/uc3_crowd.py) and the evaluation script (scripts/crowd_eval.py).

Measured on ShanghaiTech B test (316 images): native MAE 7.38 (paper 7.4); long side 1280 padded to a 1280 x 1280 square
8.95 (the fixed size Triton batches); 640: 13.3 (heads too small, undercount). So the Edge sends 1280-long images.
The network divides by 8: a 1280 x 1280 input gives a 160 x 160 density map whose sum is the people count.
"""
from __future__ import annotations

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 1280
STRIDE = 8


def prepare_square(bgr, size: int = SIZE):
    """BGR image -> (1x3xSxS float32, (h, w) of the image inside the square). Long side resized to `size`, aspect
    kept, padded with zeros after normalization (= the mean colour) at the right/bottom, as DM-Count expects."""
    import cv2
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x = np.zeros((size, size, 3), np.float32)
    x[:nh, :nw] = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (nh, nw)


def summarize(density, valid_hw, grid: int = 8) -> dict:
    """Density map of one image (S/8 x S/8, any leading 1s) -> count and a grid x grid map of people per cell over the
    image area (padding excluded); cells are row-major from the top left."""
    d = np.asarray(density, np.float32).reshape(density.shape[-2:])
    vh, vw = max(1, -(-valid_hw[0] // STRIDE)), max(1, -(-valid_hw[1] // STRIDE))
    d = d[:vh, :vw]
    rows = np.array_split(np.arange(vh), grid)
    cols = np.array_split(np.arange(vw), grid)
    cells = [[round(float(d[np.ix_(r, c)].sum()), 1) if len(r) and len(c) else 0.0 for c in cols] for r in rows]
    return {"count": round(float(d.sum()), 1), "grid": cells}
