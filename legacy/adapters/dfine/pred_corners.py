"""ORT-safe postprocess: pred_corners + pred_logits -> FDRaaS object list."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def pred_corners_to_objects(
    pred_corners,
    pred_logits,
    reg_max: int,
    top_k: int = 100,
    conf_thr: float = 0.1,
    ref_points=None,
) -> Tuple[List[dict], List[Tuple[float, float, float, float]]]:
    """
    pred_corners: (1, L, 4*(reg_max+1)), pred_logits: (1, L, num_classes).
    Works with numpy ORT outputs (no torch required).
    """
    corners = np.asarray(pred_corners, dtype=np.float32)[0]
    logits = np.asarray(pred_logits, dtype=np.float32)[0]
    bins = reg_max + 1

    probs = np.exp(logits) / (np.exp(logits).sum(axis=-1, keepdims=True) + 1e-8)
    conf = probs.max(axis=-1)
    class_id = probs.argmax(axis=-1)
    idx = np.argsort(-conf)[:top_k]

    ref_points_arr = None
    if ref_points is not None:
        ref_points_arr = np.asarray(ref_points, dtype=np.float32)[0]

    objects = []
    ref_pts_out = []
    for i in idx:
        if conf[i] < conf_thr:
            continue
        block = corners[i].reshape(4, bins)
        objects.append(
            {
                "class_id": int(class_id[i]),
                "confidence": float(conf[i]),
                "left_dist": block[0].tolist(),
                "top_dist": block[1].tolist(),
                "right_dist": block[2].tolist(),
                "bottom_dist": block[3].tolist(),
            }
        )
        if ref_points_arr is not None:
            rp = ref_points_arr[i]
            ref_pts_out.append((float(rp[0]), float(rp[1]), float(rp[2]), float(rp[3])))
        else:
            ref_pts_out.append((0.5, 0.5, 0.1, 0.1))

    return objects, ref_pts_out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)


def _peak_dist(bins: int, center: int | None = None) -> list:
    c = bins // 2 if center is None else int(np.clip(center, 0, bins - 1))
    d = [1e-6] * bins
    d[c] = 1.0
    return d


def pred_boxes_to_objects(
    pred_boxes,
    pred_logits,
    reg_max: int,
    top_k: int = 100,
    conf_thr: float = 0.1,
) -> Tuple[List[dict], List[Tuple[float, float, float, float]]]:
    """
    Fallback when ONNX exports pred_boxes (xyxy norm) instead of pred_corners.
    Builds FDRaaS 0x02 objects using real class/conf from logits + box geometry.
    """
    boxes = np.asarray(pred_boxes, dtype=np.float32)[0]
    logits = np.asarray(pred_logits, dtype=np.float32)[0]
    bins = reg_max + 1

    probs = _softmax(logits)
    conf = probs.max(axis=-1)
    class_id = probs.argmax(axis=-1)
    idx = np.argsort(-conf)[:top_k]

    objects = []
    ref_pts_out = []
    for i in idx:
        if conf[i] < conf_thr:
            continue
        x1, y1, x2, y2 = boxes[i]
        if x2 < x1 or y2 < y1:
            cx, cy, w, h = float(x1), float(y1), float(x2), float(y2)
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)
        w = float(max(x2 - x1, 0.02))
        h = float(max(y2 - y1, 0.02))
        ref_pts_out.append((cx, cy, w, h))
        peak = _peak_dist(bins)
        objects.append(
            {
                "class_id": int(class_id[i]),
                "confidence": float(conf[i]),
                "left_dist": peak,
                "top_dist": peak,
                "right_dist": peak,
                "bottom_dist": peak,
            }
        )

    return objects, ref_pts_out
