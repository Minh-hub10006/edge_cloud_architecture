"""Calibration / test inputs for the Edge half of a cut Cloud detector (plan 4/11, phase 1).

Same preprocessing as the Cloud model (inference/cloud_detr.py, cv2 path): resize to 640x640 with INTER_AREA,
BGR -> RGB, / 255, float32. Written NHWC (the layout snpe-onnx-to-dlc gives the converted input), one .raw per
image, plus input_list.txt (absolute paths) and calibration.json for scripts/convert_cut_head_snpe.sh.

  python cut_head_raw.py IMAGE_DIR OUT_DIR [--size 640]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def to_raw(path, size):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"unreadable image: {path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.float32) / 255.0  # HWC = NHWC with batch 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)  # never mix two calibration sets
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".png"))
    lines = []
    for p in paths:
        x = to_raw(p, a.size)
        assert x.shape == (a.size, a.size, 3) and 0 <= x.min() and x.max() <= 1
        dst = out / (p.stem + ".raw")
        x.tofile(dst)
        lines.append(str(dst))
    (out / "input_list.txt").write_text("\n".join(lines) + "\n")
    (out / "calibration.json").write_text(json.dumps({"input_layout": "nhwc", "size": a.size, "count": len(lines),
                                                      "preprocess": "cv2 INTER_AREA, RGB, /255"}))
    print(f"{len(lines)} raw files -> {out}")


if __name__ == "__main__":
    main()
