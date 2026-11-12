"""Path B tail that takes the wire tensor as is (5/11): uint8 + per-frame (min, max), dequantized on the GPU.

  python -B scripts/tail_uint8.py TAIL.onnx OUT.onnx [--check HEAD.onnx FRAME.jpg ...]

Prepends Cast -> * (max - min) / 255 -> + min (the same as contracts/cut_tensor.unpack) to the float tail, with new
inputs cut_u8 uint8 [N,C,H,W] and cut_range float32 [N,2] = (min, max) per frame, so the Cloud CPU only runs zstd
and copies 4x fewer bytes to Triton. --check: boxes of the new tail vs the float tail on real frames (must match).
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def convert(src, dst):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name for i in g.initializer}
    (cut,) = [i for i in g.input if i.name != "orig_target_sizes" and i.name not in inits]
    dims = [d.dim_param or d.dim_value for d in cut.type.tensor_type.shape.dim]
    g.input.remove(cut)
    g.input.insert(0, helper.make_tensor_value_info("cut_u8", TensorProto.UINT8, ["N"] + dims[1:]))
    g.input.insert(1, helper.make_tensor_value_info("cut_range", TensorProto.FLOAT, ["N", 2]))
    c = lambda name, v, dt=np.int64: numpy_helper.from_array(np.array(v, dtype=dt), name)  # noqa: E731
    g.initializer.extend([c("u8_s0", [0]), c("u8_s1", [1]), c("u8_s2", [2]), c("u8_ax", [1]),
                          c("u8_255", 255.0, np.float32), c("u8_shape", [-1, 1, 1, 1])])
    nodes = [
        helper.make_node("Cast", ["cut_u8"], ["u8_f"], to=TensorProto.FLOAT),
        helper.make_node("Slice", ["cut_range", "u8_s0", "u8_s1", "u8_ax"], ["u8_lo"]),
        helper.make_node("Slice", ["cut_range", "u8_s1", "u8_s2", "u8_ax"], ["u8_hi"]),
        helper.make_node("Sub", ["u8_hi", "u8_lo"], ["u8_d"]),
        helper.make_node("Div", ["u8_d", "u8_255"], ["u8_sc"]),
        helper.make_node("Reshape", ["u8_sc", "u8_shape"], ["u8_sc4"]),
        helper.make_node("Reshape", ["u8_lo", "u8_shape"], ["u8_lo4"]),
        helper.make_node("Mul", ["u8_f", "u8_sc4"], ["u8_m"]),
        helper.make_node("Add", ["u8_m", "u8_lo4"], [cut.name]),
    ]
    for k, n in enumerate(nodes):
        g.node.insert(k, n)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def check(tail_f, tail_u8, head, frames):
    import cv2
    import onnxruntime as ort
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contracts.cut_tensor import pack, unpack, unpack_u8
    cpu = ["CPUExecutionProvider"]
    h, tf, tu = (ort.InferenceSession(p, providers=cpu) for p in (head, tail_f, tail_u8))
    cut = [i.name for i in tf.get_inputs() if i.name != "orig_target_sizes"][0]
    worst = 0.0
    for p in frames:
        bgr = cv2.imread(p)
        x = cv2.dnn.blobFromImage(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), 1 / 255.0, swapRB=True)
        size = np.array([[bgr.shape[1], bgr.shape[0]]], np.int64)
        data = pack(h.run(None, {"images": x})[0], width=bgr.shape[1], height=bgr.shape[0], head_id="chk")
        t, _, _ = unpack(data, expected_head_id="chk")
        q, lo, hi, _, _ = unpack_u8(data, expected_head_id="chk")
        a = tf.run(["scores", "boxes"], {cut: t, "orig_target_sizes": size})
        b = tu.run(["scores", "boxes"], {"cut_u8": q, "cut_range": np.array([[lo, hi]], np.float32), "orig_target_sizes": size})
        keep_a, keep_b = a[0][0] >= .6, b[0][0] >= .6
        assert (keep_a == keep_b).all(), f"{p}: kept boxes differ"
        worst = max(worst, float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1][0][keep_a] - b[1][0][keep_b]).max(initial=0)))
        print(f"{Path(p).name}: {int(keep_a.sum())} boxes identical")
    print(f"max |score or box px| diff {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tail")
    ap.add_argument("out")
    ap.add_argument("--check", nargs="+", metavar=("HEAD", "FRAME"))
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"Refusing to overwrite {a.out}")
    convert(a.tail, a.out)
    print("wrote", a.out)
    if a.check:
        check(a.tail, a.out, a.check[0], a.check[1:])


if __name__ == "__main__":
    main()
