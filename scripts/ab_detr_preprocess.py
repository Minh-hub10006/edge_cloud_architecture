"""A/B output agreement of DETR preprocess: baseline PIL vs cv2, same model and frames.

Run on Cloud249 from a release lane:
  python -B scripts/ab_detr_preprocess.py rtdetrv2|lwdetr [--frames 64]
Gate: both directions match >= 0.95 of boxes (same class, IoU >= 0.5) and median IoU >= 0.95.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

MODELS = {
    "rtdetrv2": ("rtdetrv2_small", "/home/stu0/cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx"),
    "lwdetr": ("lwdetr_large", "/home/stu0/cloud_artifacts/lwdetr_large_candidate_20261006/lwdetr_large_coco.onnx"),
}


def iou(a, b):
    w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = w * h
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
    return inter / union if union > 0 else 0.0


def match(reference, candidate):
    """Greedy same-class matching at IoU >= 0.5; returns (iou, score_diff) pairs."""
    edges = sorted(((iou(a, b), i, j) for i, a in enumerate(reference) for j, b in enumerate(candidate)
                    if a.class_id == b.class_id), reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for overlap, i, j in edges:
        if overlap < .5:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((overlap, abs(reference[i].confidence - candidate[j].confidence)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--video", default="/home/stu0/bench_videos/traffic_1min.mp4")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    import cv2
    kind, onnx = MODELS[args.model]
    os.environ["RTDETRV2_ALLOW_IMPORTED_CANDIDATE" if kind == "rtdetrv2_small" else "LWDETR_ALLOW_IMPORTED_CANDIDATE"] = "1"
    from inference.cloud_detr import RtdetrV2SmallAdapter, LwDetrLargeAdapter
    cls = RtdetrV2SmallAdapter if kind == "rtdetrv2_small" else LwDetrLargeAdapter
    adapter = cls(onnx, confidence=args.confidence, allow_imported_candidate=True)

    capture = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise SystemExit("no frames decoded")

    results, prepare_ms = {}, {}
    for mode in ("pil", "cv2"):
        adapter.preprocess = mode
        for frame in frames[:3]:  # warm caches/CUDA before timing
            adapter.infer(frame)
        boxes, times = [], []
        for frame in frames:
            started = time.perf_counter()
            adapter.prepare(frame)
            times.append((time.perf_counter() - started) * 1000)
            boxes.append(adapter.infer(frame)[0])
        results[mode], prepare_ms[mode] = boxes, statistics.median(times)

    ref_total = sum(map(len, results["pil"]))
    new_total = sum(map(len, results["cv2"]))
    pairs = [p for a, b in zip(results["pil"], results["cv2"]) for p in match(a, b)]
    recall = len(pairs) / ref_total if ref_total else 1.0
    precision = len(pairs) / new_total if new_total else 1.0
    median_iou = statistics.median(p[0] for p in pairs) if pairs else 0.0
    score_p95 = float(np.percentile([p[1] for p in pairs], 95)) if pairs else None
    passed = recall >= .95 and precision >= .95 and median_iou >= .95
    report = {"model": args.model, "model_id": adapter.model_id, "frames": len(frames),
              "confidence": args.confidence, "boxes_pil": ref_total, "boxes_cv2": new_total,
              "matched": len(pairs), "pil_boxes_kept": recall, "cv2_boxes_matched": precision,
              "median_iou": median_iou, "score_abs_diff_p95": score_p95,
              "prepare_ms_p50": prepare_ms, "gate": "PASS" if passed else "FAIL",
              "thresholds": {"kept": .95, "matched": .95, "median_iou": .95},
              "note": "Agreement between two preprocess variants of the same model; not accuracy/mAP."}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
