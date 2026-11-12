"""P2 test level 2 on Cloud249 (10/11): DM-Count served by Triton gives the counts the laptop measured, and how fast.

  python -B scripts/crowd_triton_check.py IMAGES_DIR EXPECTED.json [--model crowd_density] [--seconds 20]
  python -B scripts/crowd_triton_check.py IMAGES_DIR EXPECTED.json --load 8 --seconds 90   # slow-lane load only

1. Parity: every image in IMAGES_DIR through Triton (inference.crowd_density + cloud_core.uc3_crowd.run_model, the
   production path) vs EXPECTED.json {image: count from the ONNX on the laptop}: difference per image.
2. Speed: 1, 4 and 8 concurrent callers for `seconds` each: images/s and latency p50/p95 per call (Triton batches
   concurrent calls, up to 4).
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("expected")
    ap.add_argument("--model", default="crowd_density")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--load", type=int, default=0, help="only keep this many callers busy for --seconds (test level 3)")
    ap.add_argument("--rate", type=float, default=0, help="with --load: at most this many images per second in total "
                    "(a realistic slow lane, e.g. 2 = 10 dense cameras at 1 frame / 5 s); 0 = as fast as possible")
    a = ap.parse_args()
    import os
    os.environ["UC3_TRITON_MODEL"] = a.model
    from cloud_core import uc3_crowd
    from inference.crowd_density import prepare_square, summarize
    expected = json.loads(Path(a.expected).read_text())
    inputs = {}
    for name, want in expected.items():
        x, valid = prepare_square(cv2.imread(str(Path(a.images, name))))
        got = summarize(uc3_crowd.run_model(x), valid)["count"]
        inputs[name] = x
        if a.load:
            continue
        print(f"== parity {name}: Triton {got:7.1f} | laptop ONNX {want:7.1f} | diff {got - want:+.2f}", flush=True)
    xs = list(inputs.values())
    for workers in ((a.load,) if a.load else (1, 4, 8)):
        lat, n, end = [], 0, time.perf_counter() + a.seconds

        start = time.perf_counter()

        def call(i):
            if a.rate:
                time.sleep(max(0.0, start + i / a.rate - time.perf_counter()))
            t = time.perf_counter()
            uc3_crowd.run_model(xs[i % len(xs)])
            return (time.perf_counter() - t) * 1000
        with ThreadPoolExecutor(workers) as pool:
            while time.perf_counter() < end:
                lat += list(pool.map(call, range(n, n + workers)))
                n += workers
        q = statistics.quantiles(lat, n=20)
        print(f"== speed {workers} callers: {n / a.seconds:.1f} images/s, per call p50 {statistics.median(lat):.0f} ms "
              f"p95 {q[18]:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
