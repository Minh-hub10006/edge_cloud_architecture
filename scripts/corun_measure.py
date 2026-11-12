"""D-FINE-X detection and plate OCR (EasyOCR) on one GPU, as two processes (the Cloud of the platform): what each
costs alone and how much each slows the other. Calibrates drl/sim_v3 (ocr_ms_per_crop, downstream_interference,
the queueing stretch) for the shared-Cloud scenario.

  python -B scripts/corun_measure.py --dfine ~/cloud_artifacts/dfine_x_candidate_20261021/dfine_x_obj2coco.onnx \
      --det-python ~/miniconda3/envs/qnn/bin/python --ocr-python ~/miniconda3/bin/python \
      [--crops DIR_OF_PLATE_JPGS] [--seconds 15] [--out ~/corun.jsonl]

Each side runs in its own interpreter (249: ONNX Runtime CUDA works in the qnn env, EasyOCR lives in the base one),
started as `--worker det|ocr`; it prints READY after warm-up, starts on GO from stdin and prints one JSON result line.

Detection: ONNX Runtime CUDA, batch 1, random 640x640 input (as scripts/onnx_capacity.py). OCR: easyocr.Reader(['en',
'vi'], gpu=True).readtext exactly as plate_lisence/comm/cloud/server.py SendPlate (x4 resize under 400 px, width_ths
.5, allowlist), on real crops from --crops or synthetic VN plates (1- and 2-line). Rate 0 = closed loop (as fast as
possible); a rate > 0 = Poisson arrivals, latency = finish - arrival (includes waiting when behind). Phases: each alone
closed loop, each alone at fixed rates, every pair of rates together, both closed loop. One JSON line per phase.
Thermal guard: stop a phase at 85 C, wait for 75 C before the next.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import os
import subprocess
import time

import numpy as np

ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-"  # plate_lisence/plate_utils.py
HOT, COOL = 85, 75


def synthetic_crops(n=40, seed=0):
    """VN-style plates at camera crop size: 1-line cars (~130x34) and 2-line motorbikes (~80x60), JPEG q92."""
    import cv2
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        two = i % 2 == 1
        w, h = (int(rng.integers(70, 95)), int(rng.integers(55, 70))) if two else (int(rng.integers(110, 160)), int(rng.integers(28, 40)))
        img = np.full((h, w, 3), int(rng.integers(200, 250)), np.uint8)
        d = lambda k: "".join(str(x) for x in rng.integers(0, 10, k))  # noqa: E731
        lines = [f"{d(2)}-{chr(65 + int(rng.integers(0, 26)))}{d(1)}", f"{d(3)}.{d(2)}"] if two else \
            [f"{d(2)}{chr(65 + int(rng.integers(0, 26)))}-{d(3)}.{d(2)}"]
        for k, txt in enumerate(lines):
            scale = h / (60 if two else 34) * .55
            y = int(h * ((k + 1) / (len(lines) + .3)))
            cv2.putText(img, txt, (3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1 if two else 2, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return out


def worker(kind, arg, rate, seconds, ocr_home=""):
    """kind 'det' (arg = onnx path) or 'ocr' (arg = crops dir or ''): READY on stdout after warm-up, waits for a line
    on stdin, then prints its latencies as one JSON line. ocr_home set: the team's plate recognizer
    (inference.plate_ocr.PlateReader, one call = one plate read) instead of SendPlate's readtext."""
    try:
        if kind == "det":
            import onnxruntime as ort
            sess = ort.InferenceSession(arg, providers=["CUDAExecutionProvider"])
            assert "CUDAExecutionProvider" in sess.get_providers(), "CUDA provider not active"
            feed = {"images": np.random.rand(1, 3, 640, 640).astype(np.float32),
                    "orig_target_sizes": np.array([[960, 476]], dtype=np.int64)}
            call = lambda i: sess.run(None, feed)  # noqa: E731
        else:
            import cv2
            crops = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(arg, "*.jpg")))] if arg else synthetic_crops()
            crops = [c for c in crops if c is not None]
            assert crops, "no plate crops"
            if ocr_home:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from inference.plate_ocr import PlateReader
                team = PlateReader(ocr_home, passes=3, gpu=True)
                import torch
                assert torch.cuda.is_available(), "team OCR must run on the GPU here"
                call = lambda i: team.read(crops[i % len(crops)])  # noqa: E731
            else:
                import easyocr
                reader = easyocr.Reader(["en", "vi"], gpu=True, verbose=False)

                def call(i):
                    c = crops[i % len(crops)]
                    if c.shape[1] < 400:  # as SendPlate
                        c = cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                    return reader.readtext(c, width_ths=.5, paragraph=False, allowlist=ALLOWLIST)
        for i in range(10):  # warm-up
            call(i)
        print("READY", flush=True)
        sys.stdin.readline()
        rng = np.random.default_rng(1 if kind == "det" else 2)
        start = time.perf_counter()
        end, nxt, lat, svc, i = start + seconds, start, [], [], 0
        while True:
            now = time.perf_counter()
            if rate:
                if nxt > now:
                    time.sleep(nxt - now)
                arrival = nxt
                nxt += rng.exponential(1 / rate)
            else:
                arrival = now
            if arrival >= end:
                break
            t0 = time.perf_counter()
            call(i)
            t1 = time.perf_counter()
            lat.append((t1 - arrival) * 1000)
            svc.append((t1 - t0) * 1000)
            i += 1
        print(json.dumps({"n": len(lat), "rate": len(lat) / seconds, "lat_p50": float(np.percentile(lat, 50)),
                          "lat_p95": float(np.percentile(lat, 95)), "svc_p50": float(np.percentile(svc, 50)),
                          "svc_mean": float(np.mean(svc))}), flush=True)
    except Exception as e:  # report instead of hanging the phase
        print("READY", flush=True)
        print(json.dumps({"error": repr(e)}), flush=True)


def gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        util, temp = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return util, temp
    except Exception:
        return float("nan"), float("nan")


def phase(a, det_rate, ocr_rate):
    """det_rate / ocr_rate: None = not running, 0 = closed loop, > 0 = Poisson arrivals per second."""
    while gpu()[1] > COOL:
        print(f"  GPU {gpu()[1]:.0f} C > {COOL} C, cooling", flush=True)
        time.sleep(20)
    procs = {}
    for kind, py, arg, rate in (("det", a.det_python, a.dfine, det_rate), ("ocr", a.ocr_python, a.crops, ocr_rate)):
        if rate is not None:
            procs[kind] = subprocess.Popen([py, "-B", os.path.abspath(__file__), "--worker", kind, "--arg", arg,
                                            "--rate", str(rate), "--seconds", str(a.seconds),
                                            "--ocr-home", a.ocr_home],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for p in procs.values():  # model loaded and warmed up (library chatter on stdout is skipped)
        while p.stdout.readline().strip() not in ("READY", ""):
            pass
    for p in procs.values():
        p.stdin.write("GO\n")
        p.stdin.flush()
    samples, end, hot = [], time.time() + a.seconds, False
    while time.time() < end:
        util, temp = gpu()
        samples.append((util, temp))
        if temp >= HOT:
            hot = True
            print(f"  THERMAL STOP {temp:.0f} C", flush=True)
            for p in procs.values():
                p.terminate()
            break
        time.sleep(.5)
    res = {}
    for kind, p in procs.items():
        try:
            lines = p.communicate(timeout=a.seconds + 120)[0].splitlines()
        except subprocess.TimeoutExpired:
            p.kill()
            lines = []
        res[kind] = next((json.loads(x) for x in reversed(lines) if x.startswith("{")), {"error": "no result"})
    util, temp = np.array(samples).T if samples else (np.array([np.nan]), np.array([np.nan]))
    return {"det_rate_target": det_rate, "ocr_rate_target": ocr_rate, **res, "gpu_util_mean": float(np.nanmean(util)),
            "gpu_temp_max": float(np.nanmax(temp)), "thermal_stop": hot}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", choices=("det", "ocr"), help=argparse.SUPPRESS)
    p.add_argument("--arg", default="", help=argparse.SUPPRESS)
    p.add_argument("--rate", type=float, default=0, help=argparse.SUPPRESS)
    p.add_argument("--dfine", default="")
    p.add_argument("--det-python", default=sys.executable, help="interpreter with onnxruntime CUDA")
    p.add_argument("--ocr-python", default=sys.executable, help="interpreter with easyocr")
    p.add_argument("--ocr-home", default="", help="team plate OCR model folder (setup_model.py --home): use it")
    p.add_argument("--crops", default="", help="directory of real plate crop JPEGs (default: synthetic VN plates)")
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--det-rates", type=float, nargs="+", default=[30, 60, 90], help="images/s (D-FINE-X alone ~100)")
    p.add_argument("--ocr-rates", type=float, nargs="+", default=[20, 50, 100], help="plate crops/s")
    p.add_argument("--out", default=os.path.expanduser("~/corun.jsonl"))
    a = p.parse_args()
    if a.worker:
        return worker(a.worker, a.arg, a.rate, a.seconds, a.ocr_home)
    assert a.dfine, "--dfine required"
    util, temp = gpu()
    print(f"GPU now: {util:.0f}% busy, {temp:.0f} C", flush=True)
    plan = [(0, None), (None, 0)] + [(r, None) for r in a.det_rates] + [(None, r) for r in a.ocr_rates] \
        + [(d, o) for d in a.det_rates for o in a.ocr_rates] + [(0, 0)]
    with open(a.out, "a", encoding="utf-8") as fh:
        for i, (d, o) in enumerate(plan):
            row = phase(a, d, o)
            row.update(phase_index=i, seconds=a.seconds, crops="real" if a.crops else "synthetic",
                       ocr="team" if a.ocr_home else "readtext", time=time.time())
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            show = lambda k: (f"{k} {row[k]['rate']:.0f}/s p50 {row[k]['lat_p50']:.1f} p95 {row[k]['lat_p95']:.1f} ms "  # noqa: E731
                              f"(svc {row[k]['svc_p50']:.1f})" if "rate" in row.get(k, {}) else f"{k} {row.get(k, '-')}")
            print(f"[{i + 1}/{len(plan)}] det {d} ocr {o}: {show('det')} | {show('ocr')} | GPU {row['gpu_util_mean']:.0f}% "
                  f"max {row['gpu_temp_max']:.0f} C", flush=True)
            if row["thermal_stop"]:
                print("thermal stop: rerun later", flush=True)
                break


if __name__ == "__main__":
    main()
