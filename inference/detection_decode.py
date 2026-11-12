"""Model-independent COCO names and strict detector-specific postprocessing."""
from __future__ import annotations

import numpy as np
from contracts import Box

COCO_NAMES = tuple("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(","))
COCO_CATEGORY_IDS = (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90)
CATEGORY_TO_CLASS = {value: index for index, value in enumerate(COCO_CATEGORY_IDS)}


def checked(value, name):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"invalid/nonfinite {name}")
    return array


def normalize(boxes, scores, labels, *, width=1, height=1):
    boxes, scores, labels = checked(boxes, "boxes"), checked(scores, "scores"), checked(labels, "labels")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),) or labels.shape != scores.shape:
        raise ValueError("inconsistent detection output dimensions")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("scores must be probabilities, not logits")
    if not (labels == labels.astype(np.int64)).all() or not ((labels >= 0) & (labels < 80)).all():
        raise ValueError("expected contiguous COCO80 labels")
    result = []
    for coords, score, label in zip(boxes, scores, labels):
        if coords[2] < coords[0] or coords[3] < coords[1]:
            raise ValueError("inverted boxes")
        xyxy = np.clip(coords / np.array([width, height, width, height]), 0, 1)
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            index = int(label)
            result.append(Box(*map(float, xyxy), float(score), index, COCO_NAMES[index]))
    return result


def rtdetr_decode(outputs, width, height, confidence, top_k):
    labels, boxes, scores = (checked(outputs[n], n) for n in ("labels", "boxes", "scores"))
    if labels.ndim != 2 or labels.shape[0] != 1 or scores.shape != labels.shape or boxes.shape != (*labels.shape, 4):
        raise ValueError("RT-DETRv2 deploy expects labels/scores [1,Q], boxes [1,Q,4]")
    if not ((scores >= 0) & (scores <= 1)).all() or not ((labels >= 0) & (labels < 80)).all() or not (labels == labels.astype(np.int64)).all():
        raise ValueError("invalid RT-DETRv2 probabilities/contiguous labels")
    # Upstream deploy returns contiguous labels BEFORE optional sparse COCO remap.
    order = np.argsort(-scores[0], kind="stable")[:top_k]
    order = order[scores[0, order] >= confidence]
    return normalize(boxes[0, order], scores[0, order], labels[0, order], width=width, height=height)


def lwdetr_decode(outputs, confidence, top_k):
    boxes, logits = checked(outputs["dets"], "dets"), checked(outputs["labels"], "labels")
    # Official export can retain decoder layers [L,B,Q,C]; use final layer.
    if boxes.ndim == 4 and logits.ndim == 4:
        if boxes.shape[:3] != logits.shape[:3] or boxes.shape[0] < 1:
            raise ValueError("inconsistent LW-DETR decoder layers")
        boxes, logits = boxes[-1], logits[-1]
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 4 or logits.shape != (*boxes.shape[:2], 91):
        raise ValueError("LW-DETR COCO export expects batch=1 and 91 category logits")
    if (boxes[..., 2:] < 0).any():
        raise ValueError("negative LW-DETR width/height")
    probabilities = 1 / (1 + np.exp(-np.clip(logits[0].astype(np.float64), -80, 80)))
    # Match upstream query x class top-k, not per-query argmax (no NMS).
    flat = probabilities.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:top_k]
    queries, categories = np.divmod(order, 91)
    valid = np.array([int(c) in CATEGORY_TO_CLASS for c in categories]) & (flat[order] >= confidence)
    coords = boxes[0, queries[valid]].copy()
    xyxy = np.concatenate((coords[:, :2] - coords[:, 2:] / 2, coords[:, :2] + coords[:, 2:] / 2), axis=1)
    labels = np.array([CATEGORY_TO_CLASS[int(c)] for c in categories[valid]], dtype=np.int64)
    return normalize(xyxy, flat[order[valid]], labels)


def rtmdet_decode(outputs, *, width, height, scale_x, scale_y, confidence, top_k, iou=0.65):
    boxes, scores = checked(outputs["boxes"], "boxes"), checked(outputs["scores"], "scores")
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 4 or scores.shape != (*boxes.shape[:2], 80):
        raise ValueError("RTMDet raw export requires boxes [1,N,4], scores [1,N,80]")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("RTMDet scores must be sigmoid probabilities")
    queries, labels = np.where(scores[0] >= confidence)
    values = scores[0, queries, labels]
    order = np.argsort(-values, kind="stable")[:30000]  # official nms_pre
    selected = []
    while len(order) and len(selected) < top_k:
        current = order[0]
        selected.append(current)
        remaining = order[1:]
        a, b = boxes[0, queries[current]], boxes[0, queries[remaining]]
        intersection = np.prod(np.maximum(np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0), axis=1)
        areas = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
        area = np.prod(np.maximum(a[2:] - a[:2], 0))
        overlap = intersection / np.maximum(area + areas - intersection, 1e-12)
        order = remaining[(labels[remaining] != labels[current]) | (overlap <= iou)]
    indices = np.asarray(selected, dtype=np.int64)
    coords = boxes[0, queries[indices]] / np.array([scale_x, scale_y, scale_x, scale_y])
    return normalize(coords, values[indices], labels[indices], width=width, height=height)
