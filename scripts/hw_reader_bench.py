"""Hardware reader cost per V4L2 capture-io-mode on an Edge (8/11): CPU of Python and GStreamer, pixels vs OpenCV, CMA.

  python scripts/hw_reader_bench.py VIDEO.mp4 [--modes auto mmap userptr dmabuf] [--seconds 20]

Per mode: 30 frames compared with OpenCV (must be identical), then the reader paced at the file rate for `seconds`,
CPU split into this Python process and the GStreamer child. CmaFree before/after each mode (a leak shows as a drop).
Stops at the first mode whose decoder cannot start, so a broken device is not hammered.
"""
import argparse
import os
import time

import cv2
import numpy as np

from media.hw_capture import HwCapture


def cma():
    return next(int(line.split()[1]) // 1024 for line in open("/proc/meminfo") if line.startswith("CmaFree"))


def child_cpu(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--modes", nargs="+", default=["auto", "mmap", "userptr", "dmabuf"])
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    ref = cv2.VideoCapture(a.video)
    want = [ref.read()[1] for _ in range(30)]
    fps = ref.get(cv2.CAP_PROP_FPS)
    c0, t0 = time.process_time(), time.perf_counter()
    for i in range(int(a.seconds * fps)):
        ref.read()
        time.sleep(max(0.0, t0 + (i + 1) / fps - time.perf_counter()))
    print(f"OpenCV (CPU decode), paced: {(time.process_time() - c0) / (time.perf_counter() - t0):.2f} cores | CmaFree {cma()} MB",
          flush=True)
    for mode in a.modes:
        before = cma()
        try:
            cap = HwCapture(a.video, decoder="v4l2h264dec", io_mode=mode)
        except Exception as exc:
            print(f"{mode:8s}: decoder did not start: {str(exc)[:200]} -> stop", flush=True)
            return
        same = max(int(np.abs(cap.read()[1].astype(int) - w.astype(int)).max()) for w in want)
        pid, g0, c0, t0 = cap._proc.pid, child_cpu(cap._proc.pid), time.process_time(), time.perf_counter()
        n = 0
        for i in range(int(a.seconds * fps)):
            if not cap.read()[0]:
                break
            n += 1
            time.sleep(max(0.0, t0 + (i + 1) / fps - time.perf_counter()))
        wall = time.perf_counter() - t0
        py, gst = (time.process_time() - c0) / wall, (child_cpu(pid) - g0) / wall
        during = cma()
        cap.release()
        time.sleep(2)
        print(f"{mode:8s}: pixels vs OpenCV max diff {same} | Python {py:.2f} + GStreamer {gst:.2f} = {py + gst:.2f} cores "
              f"({n} frames) | CmaFree before {before}, during {during}, after {cma()} MB", flush=True)


if __name__ == "__main__":
    main()
