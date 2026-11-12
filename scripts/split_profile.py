"""Where does path B's Cloud time go? (5/11: uint8 tail did not raise the 342 img/s adapter throughput.)

  PYTHONPATH=. python scripts/split_profile.py --frames DIR --head head.onnx --rtdetr RT.onnx

1. One thread, 200 calls each: per-stage ms of the full and split adapters (infer_detailed stages).
2. Same 16 concurrent calls split over P processes x T threads (1x16, 2x8, 4x4): if img/s grows with processes,
   the limit is Python's GIL in this client process, not Triton or the GPU.
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np


def build(a):
    import onnxruntime as ort
    from contracts.cut_tensor import pack
    from inference.cloud_detr import CloudDetrAdapter, RtdetrV2SmallAdapter
    from inference.triton_detr import RtdetrSplitAdapter
    os.environ["CLOUD_RUNTIME"] = "triton"
    frames = [cv2.imread(str(p)) for p in sorted(Path(a.frames).glob("*.jpg"))]
    proto = object.__new__(CloudDetrAdapter)
    proto.kind, proto.preprocess = "rtdetrv2_small", "cv2"
    head = ort.InferenceSession(a.head, providers=["CPUExecutionProvider"])
    payloads = [pack(head.run(None, {"images": proto.prepare(f)["images"]})[0], width=f.shape[1], height=f.shape[0],
                     head_id="prof") for f in frames]
    full = RtdetrV2SmallAdapter(a.rtdetr, allow_imported_candidate=True)
    return frames, payloads, full, RtdetrSplitAdapter(head_id="prof")


def worker(a, threads, seconds, out):
    from concurrent.futures import ThreadPoolExecutor
    _, payloads, _, split = build(a)
    end = time.perf_counter() + seconds

    def run(k):
        n = 0
        while time.perf_counter() < end:
            split.infer(payloads[(k + n) % len(payloads)])
            n += 1
        return n

    with ThreadPoolExecutor(threads) as ex:
        out.put(sum(ex.map(run, range(threads))))


def main():
    ap = argparse.ArgumentParser()
    for k in ("--frames", "--head", "--rtdetr"):
        ap.add_argument(k, required=True)
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    frames, payloads, full, split = build(a)
    for name, fn, items in (("full", full.infer_detailed, frames), ("split", split.infer_detailed, payloads)):
        st = []
        for i in range(200):
            _, ms, stages = fn(items[i % len(items)])
            st.append([stages["cloud_preprocess_ms"], stages["cloud_session_call_ms"], stages["cloud_postprocess_ms"], ms])
        p, s, d, t = np.mean(st[20:], axis=0)
        print(f"{name:5s} 1 thread: prepare/unpack {p:5.2f} ms | Triton call {s:5.2f} ms | decode {d:5.2f} ms | total {t:5.2f} ms")
    ctx = mp.get_context("spawn")
    for procs, threads in ((1, 16), (2, 8), (4, 4)):
        q = ctx.Queue()
        ps = [ctx.Process(target=worker, args=(a, threads, a.seconds, q)) for _ in range(procs)]
        for p in ps:
            p.start()
        total = sum(q.get() for _ in ps)
        for p in ps:
            p.join()
        print(f"split {procs} process x {threads} threads: {total / a.seconds:6.1f} img/s")


if __name__ == "__main__":
    main()
