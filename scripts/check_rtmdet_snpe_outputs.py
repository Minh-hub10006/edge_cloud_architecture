"""Offline same-frame SNPE diagnostics; creates new evidence, never admits models."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.detection_decode import rtmdet_decode
from scripts.rebuild_rtmdet_small import CHECKPOINT_SHA256, REVISION, compare_detections


SHAPES = {"boxes": (1, 8400, 4), "scores": (1, 8400, 80)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_output(path, shape):
    path = Path(path)
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f"wrong Float32 output byte count: {path}")
    value = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite native output: {path}")
    return value


def decode(values, report):
    height, width = report["image_shape"][:2]
    ratio = min(640 / width, 640 / height)
    sx, sy = int(width * ratio + .5) / width, int(height * ratio + .5) / height
    clip_h, clip_w = report["oracle_img_shape"][:2]
    boxes = values["boxes"].copy()
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, clip_w)
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, clip_h)
    evidence = report["evidence"]
    return rtmdet_decode({"boxes": boxes, "scores": values["scores"]}, width=width,
                         height=height, scale_x=sx, scale_y=sy,
                         confidence=evidence["confidence"], top_k=evidence["top_k"],
                         iou=evidence["nms_iou"])


def raw_metrics(reference, native):
    result = {}
    active = reference["scores"][0].max(axis=1) >= .1
    for name in SHAPES:
        ref, got = reference[name], native[name]
        delta = np.abs(ref.astype(np.float64) - got)
        result[name] = {"reference_min": float(ref.min()), "reference_max": float(ref.max()),
                        "native_min": float(got.min()), "native_max": float(got.max()),
                        "absolute_error_mean": float(delta.mean()),
                        "absolute_error_p95": float(np.percentile(delta, 95)),
                        "absolute_error_max": float(delta.max()),
                        "reference_active_query_count": int(active.sum()),
                        "active_query_absolute_error_mean": float(delta[0, active].mean()) if active.any() else None}
    return result


def matching(reference, native, threshold=.5):
    """Greedy same-class IoU matches are diagnostics, NOT dataset AP/recall."""
    candidates = []
    for i, a in enumerate(reference):
        aa = np.array([a.left, a.top, a.right, a.bottom])
        for j, b in enumerate(native):
            if a.class_id != b.class_id:
                continue
            bb = np.array([b.left, b.top, b.right, b.bottom])
            inter = np.prod(np.maximum(np.minimum(aa[2:], bb[2:]) - np.maximum(aa[:2], bb[:2]), 0))
            union = np.prod(aa[2:] - aa[:2]) + np.prod(bb[2:] - bb[:2]) - inter
            iou = float(inter / max(union, 1e-12))
            if iou >= threshold:
                candidates.append((iou, i, j))
    used_a, used_b, pairs = set(), set(), []
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append({"reference_index": i, "native_index": j, "iou": iou})
    return {"iou_threshold": threshold, "same_class_matches": len(pairs),
            "unmatched_reference": len(reference) - len(pairs),
            "unmatched_native": len(native) - len(pairs), "pairs": pairs,
            "note": "FP32-reference agreement only; no ground truth, not mAP."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--smoke-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new; no overwrites")
    ref_dir = args.reference_dir.resolve(strict=True)
    report = json.loads((ref_dir / "rebuild_report.json").read_text(encoding="utf-8"))
    if (report.get("export_gate") != "PASS" or report.get("source_revision") != REVISION
            or report.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise ValueError("reference source/oracle gate mismatch")
    expected_hashes = dict(line.split(None, 1)[::-1] for line in
                           (ref_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line)
    for name in ("source.png", "reference_outputs.npz", "rebuild_report.json"):
        if expected_hashes.get(name) != digest(ref_dir / name):
            raise ValueError(f"reference checksum mismatch: {name}")
    with np.load(ref_dir / "reference_outputs.npz", allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in SHAPES}
    for name, shape in SHAPES.items():
        if reference[name].shape != shape or not np.isfinite(reference[name]).all():
            raise ValueError(f"reference tensor mismatch: {name}")
    raw_dir = args.smoke_dir / "output/Result_0"
    native = {name: native_output(raw_dir / f"{name}.raw", shape) for name, shape in SHAPES.items()}
    ref_boxes, got_boxes = decode(reference, report), decode(native, report)
    # Independent stored MMDetection-validated decoded reference prevents renderer drift.
    from contracts import Box
    compare_detections([Box(**box) for box in report["boxes"]], ref_boxes)
    try:
        parity = compare_detections(ref_boxes, got_boxes)
    except ValueError as error:
        parity = {"gate": "FAIL", "reason": str(error),
                  "note": "Strict FP32 parity failure is not a quantified mAP loss."}
    import cv2
    cv2.setNumThreads(1)
    image = cv2.imread(str(ref_dir / "source.png"))
    if image is None or list(image.shape) != report["image_shape"]:
        raise ValueError("reference source image mismatch")
    canvases = []
    args.output_dir.mkdir(parents=True)
    for name, boxes, color in (("fp32", ref_boxes, (0, 210, 0)), ("npu_int8", got_boxes, (0, 165, 255))):
        canvas = image.copy()
        h, w = canvas.shape[:2]
        for box in boxes:
            p1 = (round(box.left * w), round(box.top * h))
            p2 = (round(box.right * w), round(box.bottom * h))
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, f"{box.class_name} {box.confidence:.2f}", (p1[0], max(15, p1[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        cv2.rectangle(canvas, (0, 0), (w, 65), (25, 25, 25), -1)
        cv2.putText(canvas, f"{name.upper()} | conf={report['evidence']['confidence']} | {len(boxes)} detections",
                    (20, 43), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if not cv2.imwrite(str(args.output_dir / f"{name}_boxes.png"), canvas):
            raise OSError("overlay write failed")
        canvases.append(cv2.resize(canvas, (1280, 720)))
    if not cv2.imwrite(str(args.output_dir / "comparison.png"), np.concatenate(canvases, axis=1)):
        raise OSError("comparison write failed")
    result = {"schema_version": 1, "runtime_evidence": "user SNPE --use_dsp log; profile not decoded",
              "input_binding": "user checksum-confirmed same reference NHWC; remote input not independently fetched",
              "frame_index": report["frame_index"], "output_contract_gate": "PASS",
              "quality_status": "NOT_EVALUATED", "production_ready": False,
              "strict_fp32_parity": parity, "raw_metrics": raw_metrics(reference, native),
              "reference_detections": len(ref_boxes), "native_detections": len(got_boxes),
              "reference_classes": dict(Counter(b.class_name for b in ref_boxes)),
              "native_classes": dict(Counter(b.class_name for b in got_boxes)),
              "reference_matching": matching(ref_boxes, got_boxes),
              "native_boxes": [asdict(box) for box in got_boxes],
              "hashes": {name: digest(raw_dir / f"{name}.raw") for name in SHAPES}}
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output_dir), "reference_detections": len(ref_boxes),
                      "native_detections": len(got_boxes), "strict_fp32_parity": parity["gate"],
                      "same_class_iou50_matches": result["reference_matching"]["same_class_matches"],
                      "native_classes": result["native_classes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
