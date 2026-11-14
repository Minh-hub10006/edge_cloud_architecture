"""Path B tail for the CUT2 wire (14/11): quantized codes + one (lo, step) per channel, dequantized on the GPU.

  python -B scripts/tail_quant.py TAIL.onnx OUT.onnx [--check HEAD.onnx FRAME.jpg ...]

Prepends Cast -> * step -> + lo (= contracts/cut_tensor.unpack) to the float tail, with inputs cut_q uint8 [N,C,H,W],
cut_lo and cut_step float32 [N,C]. Serves CUT1 packets too (one range broadcast to every channel).
--check: boxes of the new tail vs the float tail fed the CPU-dequantized tensor, for 2- and 8-bit packets: identical.
"""
import argparse
import sys
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
    g.input.insert(0, helper.make_tensor_value_info("cut_q", TensorProto.UINT8, ["N"] + dims[1:]))
    g.input.insert(1, helper.make_tensor_value_info("cut_lo", TensorProto.FLOAT, ["N", dims[1]]))
    g.input.insert(2, helper.make_tensor_value_info("cut_step", TensorProto.FLOAT, ["N", dims[1]]))
    g.initializer.append(numpy_helper.from_array(np.array([0, -1, 1, 1], np.int64), "q_shape"))
    nodes = [
        helper.make_node("Cast", ["cut_q"], ["q_f"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["cut_step", "q_shape"], ["q_step4"]),
        helper.make_node("Reshape", ["cut_lo", "q_shape"], ["q_lo4"]),
        helper.make_node("Mul", ["q_f", "q_step4"], ["q_m"]),
        helper.make_node("Add", ["q_m", "q_lo4"], [cut.name]),
    ]
    for k, n in enumerate(nodes):
        g.node.insert(k, n)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def check(tail_f, tail_q, head, frames):
    import cv2
    import onnxruntime as ort
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contracts.cut_tensor import pack, unpack, unpack_codes
    cpu = ["CPUExecutionProvider"]
    h, tf, tq = (ort.InferenceSession(p, providers=cpu) for p in (head, tail_f, tail_q))
    cut = [i.name for i in tf.get_inputs() if i.name != "orig_target_sizes"][0]
    worst = 0.0
    for p in frames:
        bgr = cv2.imread(p)
        x = cv2.dnn.blobFromImage(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), 1 / 255.0, swapRB=True)
        size = np.array([[bgr.shape[1], bgr.shape[0]]], np.int64)
        t = h.run(None, {"images": x})[0]
        for bits in (2, 8):
            data = pack(t, width=bgr.shape[1], height=bgr.shape[0], head_id="chk", bits=bits)
            f, _, _ = unpack(data, expected_head_id="chk")
            q, lo, step, _, _ = unpack_codes(data, expected_head_id="chk")
            a = tf.run(["scores", "boxes"], {cut: f, "orig_target_sizes": size})
            b = tq.run(["scores", "boxes"], {"cut_q": q, "cut_lo": lo, "cut_step": step, "orig_target_sizes": size})
            ka, kb = a[0][0] >= .6, b[0][0] >= .6
            assert (ka == kb).all(), f"{p} {bits} bit: kept boxes differ"
            worst = max(worst, float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1][0][ka] - b[1][0][kb]).max(initial=0)))
            print(f"{Path(p).name} {bits} bit: {len(data) / 1024:.0f} KB, {int(ka.sum())} boxes identical")
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
