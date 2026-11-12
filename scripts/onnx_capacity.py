"""GPU model capacity of a D-FINE / DEIM style ONNX (inputs images + orig_target_sizes) per batch size.

  python -B scripts/onnx_capacity.py MODEL.onnx [--batches 1 2 4 8] [--seconds 20]

Model call only (preprocessing excluded), random 640x640 input, after warm-up. Prints one JSON line per batch:
ms per call (p50/p95) and images per second.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--name", default="", help="model name for model_compare.py --capacity")
    p.add_argument("--input-size", type=int, default=640)
    args = p.parse_args(argv)
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("CUDAExecutionProvider required")
    session = ort.InferenceSession(args.onnx, providers=["CUDAExecutionProvider"])
    for batch in args.batches:
        feed = {"images": np.random.rand(batch, 3, args.input_size, args.input_size).astype(np.float32),
                "orig_target_sizes": np.array([[960, 476]] * batch, dtype=np.int64)}
        for _ in range(10):
            session.run(None, feed)
        times, end = [], time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            start = time.perf_counter()
            session.run(None, feed)
            times.append((time.perf_counter() - start) * 1000)
        p50, p95 = np.percentile(times, [50, 95])
        print(json.dumps({"name": args.name, "onnx": args.onnx, "input_size": args.input_size, "batch": batch, "calls": len(times), "ms_p50": round(float(p50), 2),
                          "ms_p95": round(float(p95), 2), "images_per_s": round(batch * 1000 / float(p50), 1)}),
              flush=True)


if __name__ == "__main__":
    main()
