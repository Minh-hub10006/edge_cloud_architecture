"""UC1 plate detector on the Edge vs the float model (P4, 11/11). On 246 every plate crop was garbage (1-47 px patches,
35/40 at det_conf 0.472) while the float .pt found real plates on the same VN vehicles (52/86, conf ~0.83).

  python -B scripts/plate_dlc_check.py make MODEL.pt VEHICLE_DETECTOR.pt FRAMES_DIR OUT_DIR     # laptop
  python -B scripts/plate_dlc_check.py check DETECTOR(.dlc|.onnx|.pt) OUT_DIR [--raw] [--layout nhwc]   # on the Edge

make:  vehicle crops (>= 64 px) from the frames and the float plate box of each -> OUT_DIR/*.jpg + expected.json.
check: the same production code (edge_agent.plate_branch.load_plate_detector) with the Edge model on those crops:
       per crop best box + conf vs the float one (IoU), then a summary. --raw also prints the model's raw outputs
       (names, shapes, value ranges) for the first crop: a constant score points at the output decoding / layout.
       --layout nhwc: feed the DLC pixels-last (SNPE converters usually make the input NHWC; the v50 plate adapter
       always feeds NCHW) - if this one matches the float model, the DLC is fine and only the feeding was wrong.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def make(model, vehicle_model, frames, out):
    from ultralytics import YOLO
    from edge_agent.plate_branch import load_plate_detector
    veh, det, expected = YOLO(vehicle_model), load_plate_detector(model), {}
    Path(out).mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(frames).glob("*.jpg")):
        im = cv2.imread(str(f))
        for k, (x1, y1, x2, y2) in enumerate(veh(im, verbose=False, classes=[2, 3, 5, 7])[0].boxes.xyxy.cpu().numpy()
                                             .astype(int)):
            if min(x2 - x1, y2 - y1) < 64:
                continue
            name = f"{f.stem}_{k:02d}.jpg"
            v = im[y1:y2, x1:x2]
            cv2.imwrite(str(Path(out, name)), v, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ps = det(cv2.imread(str(Path(out, name))))   # the decoded JPEG, as the Edge will see it
            best = max(ps, key=lambda p: p[4]) if ps else None
            expected[name] = None if best is None else [round(float(v), 2) for v in best]
    Path(out, "expected.json").write_text(json.dumps(expected, indent=1))
    print(f"{len(expected)} vehicle crops, {sum(v is not None for v in expected.values())} with a float plate")


def check(detector, out, raw, layout=None):
    from edge_agent.plate_branch import load_plate_detector
    det = load_plate_detector(detector)
    if layout:   # the same adapter, input layout forced
        from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
        d = YOLOSnpeDetector(dlc_path=detector, variant="plate", imgsz=320, score_thresh=.3)
        d.input_layout, d._pad = layout, None

        def det(frame):
            tensor, r = d._preprocess(frame)
            outs = d.session.infer_raw(tensor)[1]
            return [(b["left"], b["top"], b["right"], b["bottom"], b["confidence"])
                    for b in (d.decode_outputs(outs, r) if outs else [])]
    expected = json.loads(Path(out, "expected.json").read_text())
    if raw:
        from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
        d = YOLOSnpeDetector(dlc_path=detector, variant="plate", imgsz=320, score_thresh=.3)
        tensor, r = d._preprocess(cv2.imread(str(Path(out, next(iter(expected))))))
        res = d.session.infer_raw(tensor)
        print("input", tensor.shape, tensor.dtype, float(tensor.min()), float(tensor.max()))
        outs = res[1] if isinstance(res, tuple) else res
        for k, v in (outs.items() if isinstance(outs, dict) else enumerate(outs)):
            v = np.asarray(v)
            print(f"output {k}: shape {v.shape} min {v.min():.4f} max {v.max():.4f} distinct {len(np.unique(v.round(4)))}")
    both = miss = extra = 0
    ious, confs = [], []
    for name, want in expected.items():
        ps = det(cv2.imread(str(Path(out, name))))
        got = max(ps, key=lambda p: p[4]) if ps else None
        if got is not None:
            confs.append(round(float(got[4]), 3))
        if want is None:
            extra += got is not None
            continue
        if got is None:
            miss += 1
            continue
        both += 1
        ious.append(iou(want, got))
    n = sum(v is not None for v in expected.values())
    print(f"{detector} [{layout or 'adapter default'}]: float plates {n}; edge found {both} (missed {miss}), on plate-less crops {extra}")
    if ious:
        print(f"  box IoU vs float: median {np.median(ious):.2f}, >= 0.5 on {sum(i >= .5 for i in ious)}/{len(ious)}")
    print(f"  confidences: {len(set(confs))} distinct, {sorted(confs)[:5]} ... {sorted(confs)[-5:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "check"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--layout", choices=["nchw", "nhwc"])
    a = ap.parse_args()
    if a.cmd == "make":
        make(*a.args)
    else:
        check(a.args[0], a.args[1], a.raw, a.layout)


if __name__ == "__main__":
    sys.exit(main())
