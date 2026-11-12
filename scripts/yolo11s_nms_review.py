"""Review saved outputs at three NMS thresholds without running inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo11s_contract import Detection, SHAPES, agreement, decode, overlay


def compare_thresholds(values, prepared, reference):
    rows = []
    for threshold in (.7, .6, .5):
        detections = decode(values, prepared, nms_iou=threshold)
        rows.append({"nms_iou": threshold, "detections": [asdict(d) for d in detections],
                     "count": len(detections), "agreement": agreement(reference, detections)})
    baseline = rows[0]["detections"]
    for row in rows:
        row["removed_from_0_7"] = [d for d in baseline if d not in row["detections"]]
        row["added_vs_0_7"] = [d for d in row["detections"] if d not in baseline]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path, help="existing Edge npu_run with edge_packet and npu_output")
    mode.add_argument("--float-packet", type=Path, help="local Float references, for control testing only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("choose a new output directory; existing results are preserved")
    frames = []
    root = (args.float_packet if args.float_packet else args.run_dir / "edge_packet").resolve(strict=True)
    for number, index in enumerate((120, 600, 1200)):
        folder = root / f"frame_{index:06d}"
        record = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        required = tuple(Detection.__dataclass_fields__)
        reference = [Detection(**{k: d[k] for k in required}) for d in record["detections"]]
        height, width = record["image_shape"][:2]
        gain = min(320 / height, 320 / width)
        prepared = SimpleNamespace(width=width, height=height, gain=gain,
                                   pad_x=round((320 - width * gain) / 2 - .1),
                                   pad_y=round((320 - height * gain) / 2 - .1))
        if args.float_packet:
            with np.load(folder / "reference_outputs.npz") as archive:
                values = {k: archive[k] for k in SHAPES}
        else:
            raw = args.run_dir / "npu_output" / f"Result_{number}"
            values = {k: np.fromfile(raw / f"{k}.raw", dtype="<f4").reshape(shape)
                      for k, shape in SHAPES.items()}
        source = cv2.imread(str(folder / "source.png"))
        if source is None:
            raise ValueError("source image missing: " + str(folder))
        rows = compare_thresholds(values, prepared, reference)
        frames.append((index, source, reference, rows))
    output.mkdir(parents=True)
    for index, source, reference, rows in frames:
        overlay(source, reference, output / f"float_{index}.png")
        for row in rows:
            name = f"frame_{index}_nms_{row['nms_iou']:.1f}.png"
            overlay(source, [Detection(**d) for d in row["detections"]], output / name)
            row["overlay"] = name
            print(json.dumps({"frame": index, "nms_iou": row["nms_iou"], "count": row["count"],
                              "matches": row["agreement"]["same_class_matches"],
                              "coverage": row["agreement"]["reference_coverage"]}), flush=True)
    report = {"diagnostic_only": True, "production_ready": False, "winner": None,
              "inference_rerun": False, "confidence": .1, "class_aware": True,
              "source": "Float control" if args.float_packet else "saved SNPE NPU outputs",
              "quality_status": "NOT_EVALUATED", "note": "Float agreement is not mAP; inspect adjacent vehicles manually. No gate or production setting changed.",
              "frames": [{"frame": i, "cases": rows} for i, _image, _ref, rows in frames]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"), "overlays": str(output), "winner": None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
