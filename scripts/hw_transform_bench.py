"""Direction 2, steps 1-3 (8/11): hardware decode + qtivtransform (hardware resize + colour) straight to 640x640.

  python scripts/hw_transform_bench.py VIDEO.mp4 [--size 640] [--seconds 20]

1. Which pipeline works on this Edge: decoder capture-io-mode (auto, dmabuf) x output format (RGB, BGR); bytes per frame.
2. CPU at the file rate (15 fps) for `seconds`: this Python process reading the small frames + the GStreamer child,
   against OpenCV CPU decode + INTER_AREA resize to the same size (what media/opencv_source + the models do today).
3. Pixels: hardware 640x640 vs OpenCV cv2.resize(INTER_AREA) of the same frames (mean / max difference, share > 8).
   They will differ (another scaler); step 4 (F1 with these frames) decides whether that matters.
CmaFree before/after each pipeline. Stops at the first sign the decoder cannot start. Read-only.
"""
import argparse
import os
import re
import subprocess
import threading
import time

import cv2
import numpy as np

from media.hw_capture import SYSTEM_ENV, gst_source, stop_gracefully, _interrupt_when_parent_dies


def cma():
    return next(int(line.split()[1]) // 1024 for line in open("/proc/meminfo") if line.startswith("CmaFree"))


def child_cpu(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def chain(video, io, fmt, size):
    return gst_source(video, "v4l2h264dec", io) + ["!", "qtivtransform", "!", f"video/x-raw,format={fmt},width={size},height={size}"]


def probe(video, io, fmt, size):
    p = subprocess.run(["gst-launch-1.0", "-e", "-v", *chain(video, io, fmt, size), "!", "identity", "silent=false",
                        "eos-after=30", "!", "fakesink", "sync=false"], env=SYSTEM_ENV, capture_output=True, text=True,
                       timeout=60, preexec_fn=_interrupt_when_parent_dies)
    log = p.stdout + p.stderr
    sizes = {int(n) for n in re.findall(r"\((\d+) bytes", log)}
    err = [line.strip() for line in log.splitlines() if "ERROR" in line or "not-negotiated" in line or "erroneous" in line]
    return p.returncode, sizes, err[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    s = a.size
    props = subprocess.run(["gst-inspect-1.0", "qtivtransform"], env=SYSTEM_ENV, capture_output=True, text=True).stdout
    names = re.findall(r"^\s{2}([a-z][a-z0-9-]+)\s+:", props, re.M)
    print(f"qtivtransform properties: {', '.join(names) or 'none printed'}", flush=True)

    # 1. which pipeline works
    found = None
    for io in ("auto", "dmabuf"):
        for fmt in ("RGB", "BGR"):
            before = cma()
            rc, sizes, err = probe(a.video, io, fmt, s)
            print(f"== io {io:7s} {fmt}: exit {rc}, buffer sizes {sorted(sizes)[:3]} (plain {s * s * 3}), "
                  f"CmaFree {before} -> {cma()} MB{', ' + ' | '.join(err)[:200] if err else ''}", flush=True)
            if rc == 0 and len(sizes) == 1 and found is None:
                found = (io, fmt, sizes.pop())
    if found is None:
        print("no working qtivtransform pipeline -> stop", flush=True)
        return
    io, fmt, nbytes = found
    row = nbytes // s  # bytes per row; 8/11 on 246: 2560 for 640 px (padded 3-byte pixels, or 4-byte RGBA pixels)
    if row * s != nbytes or row < s * 3:
        print(f"{nbytes}-byte frames are not {s} rows of at least {s * 3} bytes -> stop", flush=True)
        return
    # 8/11 on 246 (PNG check): tightly packed 640x640x3 at the start, the rest of the buffer is unused padding
    layouts = [("packed, padding at the end of the buffer", 0), ("3 bytes per pixel, rows padded", 3)] +         ([("4 bytes per pixel (RGBA)", 4)] if row >= s * 4 else [])
    print(f"== using io {io}, {fmt}, {row} bytes per row for {s} px; trying: {[name for name, _ in layouts]}", flush=True)

    # 3. pixels vs OpenCV INTER_AREA, first 30 frames
    def start():
        return subprocess.Popen(["gst-launch-1.0", "-e", "-q", *chain(a.video, io, fmt, s), "!", "fdsink", "fd=1"],
                                env=SYSTEM_ENV, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                preexec_fn=_interrupt_when_parent_dies)

    def read(proc, buf, px=3):
        view, got = memoryview(buf), 0
        while got < nbytes:
            n = proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        if px == 0:
            img = np.frombuffer(buf, np.uint8)[:s * s * 3].reshape(s, s, 3)
        else:
            img = np.frombuffer(buf, np.uint8).reshape(s, row)[:, :s * px].reshape(s, s, px)[:, :, :3]
        return img if fmt == "BGR" else img[:, :, ::-1]  # compare as BGR

    buf, best = bytearray(nbytes), None
    for name, px in layouts:
        cap = cv2.VideoCapture(a.video)
        proc, diffs = start(), []
        for _ in range(30):
            hw = read(proc, buf, px)
            ok, bgr = cap.read()
            if hw is None or not ok:
                break
            d = np.abs(hw.astype(np.int16) - cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA).astype(np.int16))
            diffs.append((float(d.mean()), int(d.max()), float((d > 8).mean())))
        stop_gracefully(proc)
        m = np.array(diffs)
        print(f"== pixels as '{name}' vs OpenCV INTER_AREA: mean diff {m[:, 0].mean():.2f}, max {int(m[:, 1].max())}, "
              f"share off by > 8: {m[:, 2].mean():.2%} ({len(m)} frames)", flush=True)
        if best is None or m[:, 0].mean() < best[0]:
            best = (m[:, 0].mean(), px, name)
    px = best[1]
    print(f"== layout: {best[2]}", flush=True)
    # one frame both ways, as images, to look at (8/11: mean diff ~56 suggests letterboxing or a channel swap)
    proc = start()
    raw = None
    view, got, buf2 = memoryview(buf), 0, buf
    while got < nbytes:
        k = proc.stdout.readinto(view[got:])
        if not k:
            break
        got += k
    stop_gracefully(proc)
    rows = np.frombuffer(buf2, np.uint8).reshape(s, row)
    ok, bgr = cv2.VideoCapture(a.video).read()
    shots = {"opencv_inter_area": cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA),
             "hw_packed": np.ascontiguousarray(np.frombuffer(buf2, np.uint8)[:s * s * 3].reshape(s, s, 3)[:, :, ::-1]),
             "hw_as_3byte": np.ascontiguousarray(rows[:, :s * 3].reshape(s, s, 3)[:, :, ::-1]),
             "hw_as_4byte": np.ascontiguousarray(rows[:, :s * 4].reshape(s, s, 4)[:, :, 2::-1])}
    out = os.path.expanduser("~/v4_runs_edge/transform_frames")
    os.makedirs(out, exist_ok=True)
    for name, img in shots.items():
        cv2.imwrite(os.path.join(out, f"{name}.png"), img)
    print(f"== frame 0 saved as PNG in {out}: {', '.join(shots)}", flush=True)

    # 2. CPU at the file rate
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
    frames = int(a.seconds * fps)
    cap = cv2.VideoCapture(a.video)
    c0, t0 = time.process_time(), time.perf_counter()
    for i in range(frames):
        ok, bgr = cap.read()
        if not ok:
            break
        cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA)
        time.sleep(max(0.0, t0 + (i + 1) / fps - time.perf_counter()))
    ocv = (time.process_time() - c0) / (time.perf_counter() - t0)
    before = cma()
    proc = start()
    pid, g0, c0, t0, n = proc.pid, child_cpu(proc.pid), time.process_time(), time.perf_counter(), 0
    for i in range(frames):
        if read(proc, buf, px) is None:
            break
        n += 1
        time.sleep(max(0.0, t0 + (i + 1) / fps - time.perf_counter()))
    wall = time.perf_counter() - t0
    py, gst = (time.process_time() - c0) / wall, (child_cpu(pid) - g0) / wall
    stop_gracefully(proc)
    time.sleep(2)
    print(f"== CPU per stream at {fps:.0f} fps: OpenCV decode + INTER_AREA resize {ocv:.2f} cores | hardware decode + "
          f"qtivtransform: Python {py:.2f} + GStreamer {gst:.2f} = {py + gst:.2f} cores ({n} frames) | "
          f"CmaFree {before} -> {cma()} MB", flush=True)


if __name__ == "__main__":
    main()
