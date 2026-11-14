"""Small self-contained YOLO11s320 image contract for the staged SNPE test."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

COCO_NAMES = tuple("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(","))
SIZE = 320
SHAPES = {"boxes": (1, 2100, 4), "scores": (1, 2100, 80)}
CONFIDENCE = .1
NMS_IOU = .7


@dataclass(frozen=True)
class Detection:
    left: float
    top: float
    right: float
    bottom: float
    confidence: float
    class_id: int
    class_name: str


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_inventory(root):
    root = Path(root).resolve(strict=True)
    inventory = json.loads((root / "BUNDLE_SHA256.json").read_text(encoding="utf-8"))
    if not inventory:
        raise ValueError("empty bundle inventory")
    for name, expected in inventory.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("bundle checksum/path failed: " + name)
    return root


@dataclass(frozen=True)
class Prepared:
    tensor: np.ndarray
    width: int
    height: int
    gain: float
    pad_x: int
    pad_y: int


def prepare(bgr):
    import cv2
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
        raise ValueError("expected nonempty uint8 BGR image")
    height, width = bgr.shape[:2]
    gain = min(SIZE / height, SIZE / width)
    new_w, new_h = round(width * gain), round(height * gain)
    dw, dh = (SIZE - new_w) / 2, (SIZE - new_h) / 2
    left, right, top, bottom = round(dw - .1), round(dw + .1), round(dh - .1), round(dh + .1)
    image = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return Prepared(tensor, width, height, gain,
                    round((SIZE - width * gain) / 2 - .1),
                    round((SIZE - height * gain) / 2 - .1))


def normalize(boxes, scores, labels, *, width, height):
    boxes, scores, labels = np.asarray(boxes), np.asarray(scores), np.asarray(labels)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),)
            or labels.shape != scores.shape or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all() or not np.isfinite(labels).all()):
        raise ValueError("invalid YOLO11 detections")
    if not ((scores >= 0) & (scores <= 1)).all() or not (labels == labels.astype(np.int64)).all():
        raise ValueError("YOLO scores must be probabilities and labels integral")
    result = []
    for coords, score, label in zip(boxes, scores, labels):
        if coords[2] < coords[0] or coords[3] < coords[1] or not 0 <= int(label) < len(COCO_NAMES):
            raise ValueError("invalid YOLO box/class")
        xyxy = np.clip(coords / np.array([width, height, width, height]), 0, 1)
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            cls = int(label)
            result.append(Detection(*map(float, xyxy), float(score), cls, COCO_NAMES[cls]))
    return result


# "python" = baseline 2026-10-10 (4 ms alone, 8 ms in pipeline), kept for A/B and rollback.
# cv2.dnn.NMSBoxesBatched was tried and dropped: same-session A/B showed it slowed the
# following cv2.resize preprocess by 2.3 ms and differed on exact IoU=0.5 ties (float64).
NMS_IMPL = os.environ.get("YOLO11S_NMS", "matrix")
if NMS_IMPL not in {"matrix", "python"}:
    raise ValueError("YOLO11S_NMS must be matrix or python")
_MATRIX_MAX_CANDIDATES = 256  # cost grows with n^2; above this the per-kept-box loop is cheaper


def nms_python(boxes, values, labels, confidence, nms_iou, limit=300):
    """Class-aware greedy NMS: score > confidence, stable by score, suppress IoU > nms_iou."""
    order = np.argsort(-values, kind="stable")
    order = order[values[order] > confidence]
    selected = []
    while len(order) and len(selected) < limit:
        current = order[0]
        selected.append(current)
        remaining = order[1:]
        a, b = boxes[current], boxes[remaining]
        inter = np.prod(np.maximum(np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0), axis=1)
        union = np.prod(a[2:] - a[:2]) + np.prod(b[:, 2:] - b[:, :2], axis=1) - inter
        overlap = inter / np.maximum(union, 1e-12)
        order = remaining[(labels[remaining] != labels[current]) | (overlap <= nms_iou)]
    return np.asarray(selected, dtype=np.int64)


def nms_matrix(boxes, values, labels, confidence, nms_iou, limit=300):
    """Same selection as nms_python, bit for bit: identical float32 IoU arithmetic,
    computed once for all candidate pairs instead of once per kept box."""
    order = np.argsort(-values, kind="stable")
    order = order[values[order] > confidence]
    if len(order) > _MATRIX_MAX_CANDIDATES:
        return nms_python(boxes, values, labels, confidence, nms_iou, limit)
    b, lab = boxes[order], labels[order]
    inter = np.prod(np.maximum(np.minimum(b[:, None, 2:], b[None, :, 2:])
                               - np.maximum(b[:, None, :2], b[None, :, :2]), 0), axis=2)
    union = np.prod(b[:, 2:] - b[:, :2], axis=1)[:, None] + np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :] - inter
    suppress = (inter / np.maximum(union, 1e-12) > nms_iou) & (lab[:, None] == lab[None, :])
    removed = np.zeros(len(order), dtype=bool)
    selected = []
    for i in range(len(order)):
        if removed[i]:
            continue
        selected.append(order[i])
        if len(selected) == limit:
            break
        removed[i + 1:] |= suppress[i, i + 1:]
    return np.asarray(selected, dtype=np.int64)


def decode(outputs, prepared, *, nms_iou=NMS_IOU, confidence=CONFIDENCE):
    if not np.isfinite(nms_iou) or not 0 < nms_iou <= 1:
        raise ValueError("NMS IoU must be in (0, 1]")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    boxes, scores = (np.asarray(outputs[name]) for name in SHAPES)
    if any(outputs[name].shape != shape for name, shape in SHAPES.items()):
        raise ValueError("YOLO11s320 output signature mismatch")
    if (any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (boxes, scores))
            or not ((scores >= 0) & (scores <= 1)).all() or (boxes[..., 2:] < boxes[..., :2]).any()):
        raise ValueError("invalid YOLO output tensor")
    labels = scores[0].argmax(axis=1)
    values = scores[0, np.arange(2100), labels]
    nms = nms_matrix if NMS_IMPL == "matrix" else nms_python
    indices = nms(boxes[0], values, labels, confidence, nms_iou)
    coords = boxes[0, indices].copy()
    coords -= np.array([prepared.pad_x, prepared.pad_y] * 2, dtype=np.float32)
    coords /= prepared.gain
    return normalize(coords, values[indices], labels[indices], width=prepared.width, height=prepared.height)


def agreement(reference, candidate):
    edges = []
    for i, a in enumerate(reference):
        aa = np.array([a.left, a.top, a.right, a.bottom])
        for j, b in enumerate(candidate):
            if a.class_id != b.class_id:
                continue
            bb = np.array([b.left, b.top, b.right, b.bottom])
            inter = np.prod(np.maximum(np.minimum(aa[2:], bb[2:]) - np.maximum(aa[:2], bb[:2]), 0))
            union = np.prod(aa[2:] - aa[:2]) + np.prod(bb[2:] - bb[:2]) - inter
            overlap = float(inter / max(union, 1e-12))
            if overlap >= .5:
                edges.append((overlap, i, j))
    used_a, used_b, pairs = set(), set(), []
    for overlap, i, j in sorted(edges, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((overlap, abs(reference[i].confidence - candidate[j].confidence)))
    coverage = len(pairs) / len(reference) if reference else 0
    precision = len(pairs) / len(candidate) if candidate else 0
    median_iou = float(np.median([p[0] for p in pairs])) if pairs else 0
    score_error = float(np.percentile([p[1] for p in pairs], 95)) if pairs else None
    passed = bool(reference and candidate and coverage >= .9 and precision >= .9 and median_iou >= .9 and score_error <= .05)
    return {"gate": "PASS" if passed else "FAIL", "reference_count": len(reference),
            "candidate_count": len(candidate), "same_class_matches": len(pairs),
            "reference_coverage": coverage, "candidate_agreement": precision,
            "median_iou": median_iou, "matched_score_error_p95": score_error,
            "thresholds": {"match_iou": .5, "coverage": .9, "agreement": .9, "median_iou": .9, "score_error_p95": .05},
            "quality_status": "NOT_EVALUATED", "note": "Same-frame Float reference only; not mAP."}


def overlay(image, boxes, output):
    import cv2
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for box in boxes:
        p1, p2 = (round(box.left * w), round(box.top * h)), (round(box.right * w), round(box.bottom * h))
        cv2.rectangle(canvas, p1, p2, (0, 220, 0), 2)
        cv2.putText(canvas, f"{box.class_name} {box.confidence:.2f}", (p1[0], max(p1[1], 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1)
    if not cv2.imwrite(str(output), canvas):
        raise OSError("cannot save overlay")
