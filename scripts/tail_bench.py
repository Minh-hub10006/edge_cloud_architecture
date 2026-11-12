"""Cloud cost of path B (plan 4/11, phase 1): the Cloud half (tail) vs the full model on the 4090, per batch size.

Same runtime settings as the live Cloud (inference/cloud_detr.py: onnxruntime CUDA, default options, numpy in/out,
so host-to-device copies are included as in production). Inputs are random (the cost does not depend on content):
images [b,3,640,640] in [0,1] for the full model, a ReLU-like non-negative tensor [b,128,80,80] for the tail.
3 repetitions x 200 timed runs after 30 warm-up runs; prints and writes CSV.

  python tail_bench.py FULL.onnx TAIL.onnx [--batches 1 2 4 8] [--out tail_bench.csv]
"""
import argparse
import csv
import subprocess
import time

import numpy as np
import onnxruntime as ort


def session(path, cpu=False):
    if cpu:  # self-test only
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    s = ort.InferenceSession(path, ort.SessionOptions(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider", "CUDA required"
    return s


def feeds(sess, b, rng):
    out = {}
    for i in sess.get_inputs():
        if i.name == "orig_target_sizes":
            out[i.name] = np.full((b, 2), 640, dtype=np.int64)
        elif i.name == "images":
            out[i.name] = rng.random((b, 3, 640, 640), dtype=np.float32)
        else:  # the cut tensor: [b, 128, 80, 80], non-negative like the ReLU output
            out[i.name] = np.abs(rng.standard_normal((b, *i.shape[1:]), dtype=np.float32))
    return out


def timed(sess, f, warm=30, runs=200):
    for _ in range(warm):
        sess.run(None, f)
    t = time.perf_counter()
    for _ in range(runs):
        sess.run(None, f)
    return (time.perf_counter() - t) / runs * 1000


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "nvidia-smi n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full")
    ap.add_argument("tail")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--cpu", action="store_true", help="self-test on a machine without CUDA")
    ap.add_argument("--out", default="tail_bench.csv")
    a = ap.parse_args()
    print("before:", gpu(), "| ort", ort.__version__)
    rng = np.random.default_rng(0)
    rows = []
    for name, path in (("full", a.full), ("tail", a.tail)):
        s = session(path, a.cpu)
        for b in a.batches:
            f = feeds(s, b, rng)
            ms = [timed(s, f, warm=min(30, a.runs), runs=a.runs) for _ in range(a.reps)]
            rows.append((name, b, float(np.mean(ms)), float(np.std(ms)), b * 1000 / float(np.mean(ms))))
            print(f"{name:4s} batch {b}: {np.mean(ms):7.2f} +- {np.std(ms):5.2f} ms/batch  ->  {rows[-1][4]:6.1f} img/s")
        del s
    print("after:", gpu())
    full = {r[1]: r[4] for r in rows if r[0] == "full"}
    for n, b, _, _, ips in rows:
        if n == "tail":
            print(f"batch {b}: tail / full throughput = {ips / full[b]:.2f}x")
    with open(a.out, "w", newline="") as fh:
        csv.writer(fh).writerows([("model", "batch", "ms_mean", "ms_std", "img_per_s")] + rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
