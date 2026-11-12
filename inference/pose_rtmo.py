"""RTMO-s pose (OpenMMLab onnx_sdk end2end ONNX, Apache-2.0) pre/post-processing, as rtmlib does (P3, 11/11).

Input 1x3x640x640 float: the BGR picture letterboxed (top-left, pad 114), not normalized (the ONNX does it).
Outputs dets (B, N, 5) [x1, y1, x2, y2, score] and keypoints (B, N, 17, 3) [x, y, score], 640-input pixels.
11/11: COCO val 500 images keypoint AP 0.686 (published 0.677); Triton = CPU within 0.2 px; RTX 4090 ~100 img/s for one
caller, 428 img/s with 8.
"""
from __future__ import annotations

import numpy as np

SIZE = 640


def letterbox(bgr, size: int = SIZE, pad: int = 114):
    """-> (1x3xSxS float32, scale r: input pixel = picture pixel x r)."""
    import cv2
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    img = np.full((size, size, 3), pad, np.uint8)
    img[:int(h * r), :int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32), r


def people(dets, keypoints, r: float, conf: float = .3, iou: float = .65) -> list:
    """One picture's outputs -> [(score, box xyxy, keypoints 17x3)] in picture pixels, NMS'd, best first."""
    import cv2
    boxes, scores, kp = dets[:, :4] / r, dets[:, 4], keypoints.copy()
    kp[..., :2] /= r
    if not len(boxes):
        return []
    keep = np.array(cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes],
                                     scores.tolist(), conf, iou)).reshape(-1)
    return sorted(((float(scores[i]), boxes[i], kp[i]) for i in keep if scores[i] >= conf), key=lambda p: -p[0])


def lying(kp, min_score: float = .3) -> bool:
    """Torso (shoulder centre -> hip centre) closer to horizontal than vertical, or the visible body wider than tall."""
    seen = kp[:, 2] > min_score
    if seen[[5, 6]].any() and seen[[11, 12]].any():
        sh = kp[[5, 6]][seen[[5, 6]], :2].mean(0)
        hip = kp[[11, 12]][seen[[11, 12]], :2].mean(0)
        dx, dy = abs(hip[0] - sh[0]), abs(hip[1] - sh[1])
        return dx > dy
    if seen.sum() >= 5:
        xs, ys = kp[seen, 0], kp[seen, 1]
        return xs.max() - xs.min() > ys.max() - ys.min()
    return False
