"""Export the team plate detector (YOLOv8n, 1 class) to the split ONNX the platform's plate adapter reads:
input `images` [1,3,S,S] RGB/255 letterboxed NCHW, outputs `boxes` [1,4,N] (cx,cy,w,h px) and `scores` [1,1,N].
Separate outputs keep independent INT8 scales when the ONNX is converted to a DLC for the NPU.

  python -B scripts/export_plate_onnx.py --weights best_320.pt --imgsz 320 --out plate_best320_split.onnx \
      --check-video vn2.mp4

--check-video: runs the exported graph (onnx reference evaluator, no onnxruntime needed) and the PyTorch model through
the platform's letterbox and decoder on a few frames and asserts both give the same boxes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export(weights, imgsz, out):
    import onnx
    from onnx import helper, TensorProto
    from ultralytics import YOLO
    tmp = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=13, simplify=False, dynamic=False)
    m = onnx.load(tmp)
    (pred,) = m.graph.output
    n = pred.type.tensor_type.shape.dim[2].dim_value
    if pred.type.tensor_type.shape.dim[1].dim_value != 5:
        raise ValueError("expected a 1-class YOLOv8 head [1,5,N]")
    if m.graph.input[0].name != "images":
        raise ValueError("expected the ultralytics input name 'images'")
    for name, value in (("s0", [0]), ("s4", [4]), ("s5", [5]), ("ax", [1])):
        m.graph.initializer.append(helper.make_tensor(name, TensorProto.INT64, [1], value))
    m.graph.node.extend([helper.make_node("Slice", [pred.name, "s0", "s4", "ax"], ["boxes"]),
                         helper.make_node("Slice", [pred.name, "s4", "s5", "ax"], ["scores"])])
    del m.graph.output[:]
    m.graph.output.extend([helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4, n]),
                           helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, n])])
    onnx.checker.check_model(m)
    onnx.save(m, out)
    os.remove(tmp)
    return out


def check(weights, imgsz, out, video, frames=3):
    import cv2
    from onnx.reference import ReferenceEvaluator
    from edge_agent.plate_branch import load_plate_detector, plate_decoder
    ref = load_plate_detector(weights, imgsz=imgsz, conf=.3)
    graph = ReferenceEvaluator(out)
    dec = plate_decoder(imgsz, .3)
    cap = cv2.VideoCapture(video)
    checked = 0
    for i in range(0, 600, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        want = ref(frame)
        if not want:
            continue
        tensor, r = dec._preprocess(frame)
        boxes, scores = graph.run(None, {"images": tensor})
        got = dec.decode_outputs({"boxes": boxes, "scores": scores}, r)
        got = sorted((b["left"], b["top"], b["right"], b["bottom"]) for b in got)
        want = sorted(b[:4] for b in want)
        print(f"frame {i}: pytorch {len(want)} split-onnx {len(got)}", flush=True)
        assert len(got) == len(want), (got, want)
        assert np.abs(np.array(got) - np.array(want)).max() < .5, (got, want)
        checked += 1
        if checked == frames:
            break
    assert checked, "no frame with a plate to check"
    print(f"OK: {checked} frames match", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--out", required=True)
    p.add_argument("--check-video", default="")
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    export(a.weights, a.imgsz, a.out)
    print("wrote", a.out, flush=True)
    if a.check_video:
        check(a.weights, a.imgsz, a.out, a.check_video)


if __name__ == "__main__":
    main()
