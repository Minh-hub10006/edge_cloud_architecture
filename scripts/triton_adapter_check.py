"""Step 3 gate on 249: the Cloud adapters through Triton (CLOUD_RUNTIME=triton) vs the live onnxruntime CUDA adapters.

  PYTHONPATH=. python scripts/triton_adapter_check.py --frames DIR --head head.onnx --head-id ID \
      --rtdetr RT.onnx --dfinex DX.onnx [--seconds 10 --threads 16]

1. Same boxes (score >= .6, IoU .5): Triton adapter vs the same ONNX on onnxruntime CPU (exact reference, same
   prepare/decode as inference/cloud_detr.py), per model; split path (float head.onnx -> contracts/cut_tensor.pack ->
   RtdetrSplitAdapter) vs the full RT-DETR reference.
2. Throughput through the adapters (preprocess + Triton + decode) with `threads` concurrent calls, like the Cloud
   service with CLOUD_MAX_INFLIGHT=16; Triton's mean executed batch from its metrics.
Stops when the GPU stays >= 450 W or >= 75 C for 3 s.
"""
import argparse
import os
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

stop = threading.Event()


def guard():
    hot = 0
    while not stop.is_set():
        w, c = map(float, subprocess.run(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True).stdout.split(","))
        hot = hot + 1 if (w >= 450 or c >= 75) else 0
        if hot >= 3:
            print(f"!! SAFETY STOP: GPU {w:.0f} W, {c:.0f} C", flush=True)
            stop.set()
        time.sleep(1)


def agreement(a, b, thr=.6):
    a = [x for x in a if x.confidence >= thr]
    b = [x for x in b if x.confidence >= thr]
    if not a and not b:
        return 1.0
    used, tp = set(), 0
    for x in sorted(a, key=lambda x: -x.confidence):
        best, j = .5, None
        for k, y in enumerate(b):
            if k in used:
                continue
            w = max(0, min(x.right, y.right) - max(x.left, y.left))
            h = max(0, min(x.bottom, y.bottom) - max(x.top, y.top))
            inter = w * h
            u = (x.right - x.left) * (x.bottom - x.top) + (y.right - y.left) * (y.bottom - y.top) - inter
            if u > 0 and inter / u >= best:
                best, j = inter / u, k
        if j is not None:
            used.add(j)
            tp += 1
    return 2 * tp / (len(a) + len(b))


def counters(model):
    text = urllib.request.urlopen("http://localhost:18002/metrics", timeout=5).read().decode()
    get = lambda n: float(re.search(rf'^{n}{{model="{model}",version="1"}} (\S+)', text, re.M).group(1))  # noqa: E731
    return get("nv_inference_count"), get("nv_inference_exec_count")


def throughput(name, triton_model, fn, items, threads, seconds):
    if stop.is_set():
        return
    n0, e0 = counters(triton_model)
    end, count, lock = time.perf_counter() + seconds, [0], threading.Lock()

    def worker(k):
        i = k
        while time.perf_counter() < end and not stop.is_set():
            fn(items[i % len(items)])
            i += threads
            with lock:
                count[0] += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(worker, range(threads)))
    dur = time.perf_counter() - t0
    n1, e1 = counters(triton_model)
    print(f"   {name:12s} {count[0] / dur:6.1f} img/s with {threads} threads | Triton mean batch {(n1 - n0) / max(e1 - e0, 1):.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--frames", "--head", "--head-id", "--rtdetr", "--dfinex"):
        ap.add_argument(k, required=True)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    import onnxruntime as ort
    from contracts.cut_tensor import pack
    from inference.cloud_detr import CloudDetrAdapter, DfineXAdapter, RtdetrV2SmallAdapter
    from inference.detection_decode import rtdetr_decode
    from inference.triton_detr import RtdetrSplitAdapter
    threading.Thread(target=guard, daemon=True).start()
    frames = [cv2.imread(str(p)) for p in sorted(Path(a.frames).glob("*.jpg"))]
    kw = dict(allow_imported_candidate=True)
    proto = object.__new__(CloudDetrAdapter)  # only prepare(): 640 INTER_AREA, RGB, /255 (no session, no manifest)
    proto.kind, proto.preprocess = "rtdetrv2_small", "cv2"

    class Ref:  # the same ONNX on CPU: exact reference for the boxes
        def __init__(self, path):
            self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

        def infer(self, f):
            names = ["labels", "boxes", "scores"]
            out = dict(zip(names, self.s.run(names, proto.prepare(f))))
            return rtdetr_decode(out, f.shape[1], f.shape[0], .10, 300), 0.0

    ref = {"rtdetr": Ref(a.rtdetr), "dfinex": Ref(a.dfinex)}
    os.environ["CLOUD_RUNTIME"] = "triton"
    tri = {}
    for key, cls, path in (("rtdetr", RtdetrV2SmallAdapter, a.rtdetr), ("dfinex", DfineXAdapter, a.dfinex)):
        try:
            tri[key] = cls(path, **kw)
        except ValueError as exc:
            print(f"== {key}: Triton adapter refused: {exc}", flush=True)
    split = RtdetrSplitAdapter(head_id=a.head_id)
    head = ort.InferenceSession(a.head, providers=["CPUExecutionProvider"])
    payloads = []
    for f in frames:
        t = head.run(None, {"images": proto.prepare(f)["images"]})[0]
        payloads.append(pack(t, width=f.shape[1], height=f.shape[0], head_id=a.head_id))
    print(f"{len(frames)} frames | path-B payload {np.mean([len(p) for p in payloads]) / 1024:.0f} KB mean", flush=True)
    for key, adapter in tri.items():
        ag = [agreement(ref[key].infer(f)[0], adapter.infer(f)[0]) for f in frames]
        print(f"== {key}: Triton adapter vs ORT CPU, box agreement worst {min(ag):.3f} {'PASS' if min(ag) >= .99 else 'FAIL'}")
    ag = [agreement(ref["rtdetr"].infer(f)[0], split.infer(p)[0]) for f, p in zip(frames, payloads)]
    print(f"== split: tensor path vs full RT-DETR (ORT CPU), box agreement worst {min(ag):.3f} mean {np.mean(ag):.3f}")
    print(f"-- throughput through the adapters, {a.seconds:.0f} s each", flush=True)
    if "rtdetr" in tri:
        throughput("rtdetr_full", "rtdetr_full", lambda f: tri["rtdetr"].infer(f), frames, a.threads, a.seconds)
    throughput("rtdetr_split", "rtdetr_tail_u8", split.infer, payloads, a.threads, a.seconds)
    if "dfinex" in tri:
        throughput("dfinex_full", "dfinex_full", lambda f: tri["dfinex"].infer(f), frames, a.threads, a.seconds)
    stop.set()


if __name__ == "__main__":
    main()
