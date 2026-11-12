"""Slow lane step 1 on a Qualcomm Edge (8/11): the full-resolution branch of media/hw_capture.HwCapture.

  python scripts/hw_hires_check.py VIDEO.mp4 [--decoder v4l2h264dec] [--every 3] [--seconds 20]

1. Pairing: on frames 0, n, 2n, ... the full-resolution frame, shrunk to 640x640 (INTER_AREA), is compared with the
   640x640 analytics frame of the same index and of the neighbour indexes: the same index must be clearly closest.
2. CPU at the file rate: Python + GStreamer, the reader alone vs the reader with the full-resolution branch.
"""
import argparse
import os
import time

import cv2
import numpy as np

from media.hw_capture import HwCapture


def child_cpu(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def cpu(video, decoder, every, seconds):
    cap = HwCapture(video, decoder=decoder, size=640, hires_every=every)
    fps, n = cap.fps, int(seconds * cap.fps)
    pid, g0, c0, t0, got = cap._proc.pid, child_cpu(cap._proc.pid), time.process_time(), time.perf_counter(), 0
    for i in range(n):
        ok, _ = cap.read()
        if not ok:
            break
        got += cap.last_hires is not None
        time.sleep(max(0.0, t0 + (i + 1) / fps - time.perf_counter()))
    wall = time.perf_counter() - t0
    py, gst = (time.process_time() - c0) / wall, (child_cpu(pid) - g0) / wall
    cap.release()
    return py, gst, got, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--decoder", default="v4l2h264dec")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()

    cap = HwCapture(a.video, decoder=a.decoder, size=640, hires_every=a.every)
    small, big = [], {}
    for i in range(31):
        ok, px = cap.read()
        if not ok:
            break
        small.append(px)
        if cap.last_hires is not None:
            big[i] = cap.last_hires
    cap.release()
    print(f"== full-resolution frames on indexes {sorted(big)} shape {next(iter(big.values())).shape if big else None}")
    same, other = [], []
    for i, hires in big.items():
        shrunk = cv2.resize(hires, (640, 640), interpolation=cv2.INTER_AREA).astype(np.int16)
        for j in range(max(0, i - 2), min(len(small), i + 3)):
            d = float(np.abs(shrunk - small[j]).mean())
            (same if j == i else other).append(d)
    print(f"== pairing: mean diff same index {np.mean(same):.2f} | neighbour indexes min {min(other):.2f} "
          f"-> {'OK' if np.mean(same) < min(other) else 'MISMATCH (branch is offset)'}", flush=True)

    for every in (0, a.every):
        py, gst, got, n = cpu(a.video, a.decoder, every, a.seconds)
        print(f"== CPU at {n / a.seconds:.0f} fps, full-res every {every or '-'}: Python {py:.2f} + GStreamer {gst:.2f} "
              f"= {py + gst:.2f} cores, full-res frames {got}/{n}", flush=True)


if __name__ == "__main__":
    main()
