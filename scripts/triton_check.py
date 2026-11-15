"""Triton gate (v4 main lane, step 2), run on 249 against the v4-triton container (gRPC 18001, metrics 18002).

  python triton_check.py --frames DIR --head head.onnx [--repo ~/triton_repo_20261105] [--no-shm]

1. Batch parity on REAL frames (DIR/*.jpg, preprocessed like inference/cloud_detr.py: 640 INTER_AREA, RGB, /255;
   the tail gets the float head's output): detections (score >= .6, greedy IoU .5) of one batch-4 request vs four
   batch-1 requests, and batch-1 vs onnxruntime CPU on the same model.onnx. PASS = agreement >= .99 on every frame.
2. Load: C concurrent "cameras" (1, 2, 4, 8, 16), each sending single frames back to back; Triton batches them
   (max 4, 8 ms). The big input goes through system shared memory (the Cloud worker runs on 249 too; the container
   must run with --ipc=host), unless --no-shm. Reports img/s, latency p50/p95 and Triton's mean executed batch.
rtdetr_tail_q (15/11, --models): the tail of the live CUT2 wire - the float head's output packed at 2 bits per value
(contracts/cut_tensor.py) and fed as codes + per-channel lo/step; parity only (its load: scripts/cloud_v4_load.py).
Safety: stops all load when the GPU stays >= 450 W or >= 75 C for 3 s (Cloud limits, power raised 5/11).
"""
import argparse
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm

MODELS = ["rtdetr_full", "rtdetr_tail", "dfinex_full"]
THR, IOU = .6, .5
LIMIT_W, LIMIT_C = 450.0, 75.0  # 5/11: user raised the Cloud power limit; 450 W = the 4090 board limit
stop_all = threading.Event()


def guard():
    hot = 0
    while not stop_all.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split(",")
        w, c = float(out[0]), float(out[1])
        hot = hot + 1 if (w >= LIMIT_W or c >= LIMIT_C) else 0
        if hot >= 3:
            print(f"!! SAFETY STOP: GPU {w:.0f} W, {c:.0f} C", flush=True)
            stop_all.set()
        time.sleep(1)


def load_frames(d):
    xs, sizes = [], []
    for p in sorted(Path(d).glob("*.jpg"))[:8]:
        bgr = cv2.imread(str(p))
        sizes.append([bgr.shape[1], bgr.shape[0]])
        x = cv2.cvtColor(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        xs.append(x.astype(np.float32).transpose(2, 0, 1) / 255)
    assert len(xs) == 8, "need 8 jpg frames"
    return np.stack(xs), np.array(sizes, dtype=np.int64)


def dets(scores, boxes):
    keep = scores >= THR
    return boxes[keep], scores[keep]


def iou(a, b):
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def agreement(x, y):
    (bx, sx), (by, _) = x, y
    if len(bx) == 0 and len(by) == 0:
        return 1.0
    used, tp = np.zeros(len(by), bool), 0
    for k in np.argsort(-sx):
        if len(by):
            v = np.where(used, -1, iou(bx[k], by))
            j = int(v.argmax())
            if v[j] >= IOU:
                used[j], tp = True, tp + 1
    return 2 * tp / (len(bx) + len(by))


def take(big, a, b):   # one input array, or a dict of them (rtdetr_tail_q), frames a..b
    return {k: v[a:b] for k, v in big.items()} if isinstance(big, dict) else big[a:b]


def feeds_for(meta, big, sizes):
    out = {}
    for i in meta.inputs:
        out[i.name] = sizes if i.name == "orig_target_sizes" else (big[i.name] if isinstance(big, dict) else big)
    return out


def cut2_inputs(t, sizes):
    """Float head outputs -> what the Edge sends at 2 bits and the worker feeds rtdetr_tail_q (codes, lo, step)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contracts.cut_tensor import pack, unpack_codes
    parts = [unpack_codes(pack(np.ascontiguousarray(t[k:k + 1]), width=int(sizes[k][0]), height=int(sizes[k][1]),
                               head_id="chk", bits=2), expected_head_id="chk")[:3] for k in range(len(t))]
    return {name: np.ascontiguousarray(np.concatenate([p[j] for p in parts])) for j, name in
            enumerate(("cut_q", "cut_lo", "cut_step"))}


def infer(client, model, feeds):
    ins = []
    for name, x in feeds.items():
        t = grpcclient.InferInput(name, list(x.shape), {np.dtype(np.int64): "INT64", np.dtype(np.uint8): "UINT8"}
                                  .get(x.dtype, "FP32"))
        t.set_data_from_numpy(x)
        ins.append(t)
    r = client.infer(model, ins)
    return r.as_numpy("scores"), r.as_numpy("boxes")


def counters(url, model):
    text = urllib.request.urlopen(url, timeout=5).read().decode()
    get = lambda name: float(re.search(rf'^{name}{{model="{model}",version="1"}} (\S+)', text, re.M).group(1))  # noqa: E731
    return get("nv_inference_count"), get("nv_inference_exec_count")


def parity(client, model, meta, big, sizes, cpu):
    worst_b4, worst_cpu, max_diff = 1.0, 1.0, 0.0
    for g in (0, 4):
        s4, b4 = infer(client, model, feeds_for(meta, take(big, g, g + 4), sizes[g:g + 4]))
        for k in range(4):
            f = feeds_for(meta, take(big, g + k, g + k + 1), sizes[g + k:g + k + 1])
            s1, b1 = infer(client, model, f)
            sc, bc = cpu.run(["scores", "boxes"], f)
            worst_b4 = min(worst_b4, agreement(dets(s4[k], b4[k]), dets(s1[0], b1[0])))
            worst_cpu = min(worst_cpu, agreement(dets(s1[0], b1[0]), dets(sc[0], bc[0])))
            max_diff = max(max_diff, float(np.abs(np.sort(s4[k]) - np.sort(s1[0])).max()))
    ok = worst_b4 >= .99
    print(f"== {model}: box agreement batch4 vs batch1 (worst frame) {worst_b4:.3f} {'PASS' if ok else 'FAIL'} | "
          f"Triton GPU vs ORT CPU {worst_cpu:.3f} | max score diff b4/b1 {max_diff:.1e}", flush=True)


def load(a, model, meta, big, sizes):
    big_name = next(i.name for i in meta.inputs if i.name != "orig_target_sizes")
    for c in a.conc:
        if stop_all.is_set():
            return
        handles, clients = [], [grpcclient.InferenceServerClient(a.url) for _ in range(c)]
        per_cam = []
        for k in range(c):
            x = np.ascontiguousarray(big[k % 8:k % 8 + 1])
            ins = [grpcclient.InferInput(big_name, list(x.shape), "FP32"),
                   grpcclient.InferInput("orig_target_sizes", [1, 2], "INT64")]
            ins[1].set_data_from_numpy(sizes[k % 8:k % 8 + 1])
            if a.no_shm:
                ins[0].set_data_from_numpy(x)
            else:
                region = f"v4cam{k}_{model}"
                h = shm.create_shared_memory_region(region, f"/{region}", x.nbytes)
                shm.set_shared_memory_region(h, [x])
                clients[k].register_system_shared_memory(region, f"/{region}", x.nbytes)
                ins[0].set_shared_memory(region, x.nbytes)
                handles.append((clients[k], region, h))
            per_cam.append(ins)
        n0, e0 = counters(a.metrics, model)
        end = time.perf_counter() + a.seconds

        def camera(k):
            lat = []
            while time.perf_counter() < end and not stop_all.is_set():
                t = time.perf_counter()
                clients[k].infer(model, per_cam[k])
                lat.append(time.perf_counter() - t)
            return lat

        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            lat = np.concatenate([np.array(x) for x in ex.map(camera, range(c))]) * 1000
        dur = time.perf_counter() - t0
        n1, e1 = counters(a.metrics, model)
        for cl, region, h in handles:
            cl.unregister_system_shared_memory(region)
            shm.destroy_shared_memory_region(h)
        for cl in clients:
            cl.close()
        print(f"   {c:2d} cameras: {len(lat) / dur:6.1f} img/s  latency p50 {np.percentile(lat, 50):5.1f} "
              f"p95 {np.percentile(lat, 95):5.1f} ms  mean batch run {(n1 - n0) / max(e1 - e0, 1):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--head", required=True, help="head.onnx of the RT-DETR cut (makes the tail's real input)")
    ap.add_argument("--repo", default=str(Path.home() / "triton_repo_20261105"))
    ap.add_argument("--url", default="localhost:18001")
    ap.add_argument("--metrics", default="http://localhost:18002/metrics")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--no-shm", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS, help=f"default {MODELS}; also rtdetr_tail_q")
    a = ap.parse_args()
    threading.Thread(target=guard, daemon=True).start()
    x, sizes = load_frames(a.frames)
    cpu = {"providers": ["CPUExecutionProvider"]}
    head = ort.InferenceSession(a.head, **cpu)
    t = np.concatenate([head.run(None, {"images": x[k:k + 1]})[0] for k in range(8)])
    client = grpcclient.InferenceServerClient(a.url)
    inputs = {"rtdetr_tail": t, "rtdetr_tail_q": cut2_inputs(t, sizes)}
    for model in a.models:
        meta = client.get_model_metadata(model)
        parity(client, model, meta, inputs.get(model, x), sizes,
               ort.InferenceSession(f"{a.repo}/{model}/1/model.onnx", **cpu))
    print(f"-- load ({'gRPC copy' if a.no_shm else 'system shared memory'}), {a.seconds:.0f} s per step", flush=True)
    for model in [m for m in a.models if m != "rtdetr_tail_q"]:
        print(f"== {model}", flush=True)
        load(a, model, client.get_model_metadata(model), inputs.get(model, x), sizes)
    stop_all.set()


if __name__ == "__main__":
    main()
