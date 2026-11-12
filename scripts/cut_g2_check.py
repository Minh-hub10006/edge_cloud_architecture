"""Gate G2 (plan 4/11): does the NPU head tensor match the float head, and do the tail's boxes stay the same?

Inputs: head.onnx / tail.onnx (scripts/cut_points.py), the NHWC .raw inputs given to snpe-net-run (in its
test_list.txt order), and snpe-net-run's output dir (Result_<i>/**/*.raw, float32 NHWC).
Pass: correlation >= .99 on every frame.

  python -B scripts/cut_g2_check.py HEAD.onnx TAIL.onnx TEST_LIST NPU_OUT_DIR [--size 640]
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cut_points import agreement, detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head")
    ap.add_argument("tail")
    ap.add_argument("test_list", help="test_list.txt from the NPU run (order of Result_<i>)")
    ap.add_argument("npu_out")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--raw-dir", help="local folder holding the same .raw files (default: next to test_list)")
    a = ap.parse_args()
    so = {"providers": ["CPUExecutionProvider"]}
    head, tail = ort.InferenceSession(a.head, **so), ort.InferenceSession(a.tail, **so)
    cut = head.get_outputs()[0].name
    outs = [o.name for o in tail.get_outputs()]
    others = [i.name for i in tail.get_inputs() if i.name != cut]
    raw_dir = Path(a.raw_dir or Path(a.test_list).parent / "test_raw")
    names = [Path(line.strip()).name for line in open(a.test_list) if line.strip()]
    s = a.size
    rows = []
    for i, name in enumerate(names):
        x = np.fromfile(raw_dir / name, dtype=np.float32).reshape(1, s, s, 3).transpose(0, 3, 1, 2)
        ref = head.run(None, {"images": np.ascontiguousarray(x)})[0]
        (npu_file,) = list((Path(a.npu_out) / f"Result_{i}").rglob("*.raw"))
        npu = np.fromfile(npu_file, dtype=np.float32).reshape(ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[1])
        npu = npu.transpose(0, 3, 1, 2)
        corr = float(np.corrcoef(ref.ravel(), npu.ravel())[0, 1])
        rel = float(np.abs(ref - npu).mean() / max(np.abs(ref).mean(), 1e-12))
        feeds = {o: np.array([[s, s]], dtype=np.int64) for o in others}
        det_ref = detections(tail.run(None, {cut: ref} | feeds), outs)
        det_npu = detections(tail.run(None, {cut: np.ascontiguousarray(npu)} | feeds), outs)
        rows.append((name, corr, rel, agreement(det_ref, det_npu)))
        print(f"{name:40s} corr {corr:.4f}  rel_mae {rel:.4f}  box agreement {rows[-1][3]:.3f}")
    c = np.array([r[1] for r in rows])
    print(f"\n{len(rows)} frames | corr min {c.min():.4f} mean {c.mean():.4f} | rel_mae mean {np.mean([r[2] for r in rows]):.4f}"
          f" | box agreement mean {np.mean([r[3] for r in rows]):.3f} | G2 {'PASS' if c.min() >= .99 else 'FAIL'}")


if __name__ == "__main__":
    main()
