"""A/B YOLO11s NMS on the same real NPU outputs: Python greedy vs numpy matrix.

Run on Edge246 from a release lane with the cache-off daemon env (see scripts/models.sh):
  python -B scripts/ab_yolo11s_nms.py [--frames 64]
Gate: every frame selects identical boxes in identical order.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

DLC = ("/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/"
       "npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc")


def iou64(a, b):
    w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = w * h
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
    return inter / union if union > 0 else 0.0


def explain(python_boxes, cv2_boxes):
    """For each box kept by only one side, its closest same-class IoU to the other side's boxes."""
    rows = []
    for kept_by, mine, other in (("python_only", python_boxes, cv2_boxes), ("matrix_only", cv2_boxes, python_boxes)):
        for box in mine:
            if box in other:
                continue
            overlaps = [iou64(box, o) for o in other if o.class_id == box.class_id]
            nearest = min(overlaps, key=lambda v: abs(v - .5), default=None)
            rows.append({"kept_by": kept_by, "class": box.class_name, "score": round(box.confidence, 4),
                         "iou_nearest_to_0.5": nearest,
                         "abs_iou_minus_0.5": abs(nearest - .5) if nearest is not None else 1.0})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--video", default="/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4")
    args = parser.parse_args()
    import cv2
    import yolo11s_contract as contract
    from inference.yolo11s_snpe import Yolo11sSnpeAdapter
    adapter = Yolo11sSnpeAdapter(DLC, allow_diagnostic=True, confidence=.1, nms_iou=.5)
    capture = cv2.VideoCapture(args.video)
    identical, boxes_total, times = 0, 0, {"python": [], "matrix": []}
    frames = 0
    diffs, candidates = [], []
    while frames < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        prepared = adapter.prepare_frame(frame)
        _, outputs = adapter.session.infer_raw(prepared.tensor)
        shaped = {name: np.asarray(outputs[name]).reshape(shape) for name, shape in contract.SHAPES.items()}
        candidates.append(int((shaped["scores"][0].max(axis=1) > .1).sum()))
        results = {}
        for impl in ("python", "matrix"):
            contract.NMS_IMPL = impl
            started = time.perf_counter()
            results[impl] = contract.decode(shaped, prepared.geometry, nms_iou=.5, confidence=.1)
            times[impl].append((time.perf_counter() - started) * 1000)
        same = results["python"] == results["matrix"]
        identical += same
        if not same:
            diffs.extend(explain(results["python"], results["matrix"]))
        boxes_total += len(results["python"])
        frames += 1
    capture.release()
    report = {"frames": frames, "identical_frames": identical, "boxes_python": boxes_total,
              "decode_ms_p50": {k: statistics.median(v) for k, v in times.items()},
              "candidates_above_conf": {"p50": statistics.median(candidates), "max": max(candidates)},
              "differing_boxes": len(diffs),
              "differing_boxes_max_abs_iou_minus_threshold": max((d["abs_iou_minus_0.5"] for d in diffs), default=None),
              "differences": diffs[:20],
              "gate": "PASS" if frames and identical == frames else "FAIL",  # same float32 math: exact
              "note": "Same NPU outputs decoded twice; decode_ms includes NMS and box normalization."}
    print(json.dumps(report, indent=2))
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
