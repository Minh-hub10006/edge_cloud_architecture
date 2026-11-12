"""Is hardware decode worth it on a Qualcomm Edge (7/11)? CPU cost and pixel parity vs today's OpenCV decode.

  python -B scripts/hw_decode_check.py VIDEO.mp4 [--decoder v4l2h264dec|qtic2vdec] [--frames 30]

1. CPU at the real camera pace (15 fps, whole file): OpenCV (CPU, as media/opencv_source.py) vs GStreamer hardware decode
   (runbook QUALCOMM_EDGE_246_EDGE_252_RTSP_HARDWARE_CODEC: 246 v4l2h264dec, 252 qtic2vdec), plus the NV12->BGR
   conversion Python would still do. Reported as CPU cores busy (1.00 = one core).
2. Hardware decode speed when not paced (frames/s).
3. Hardware decode + GStreamer videoconvert to BGR (handles the decoder's padded layout), paced.
4. Pixel parity: the first frames from both paths, BGR, compared (mean / max abs difference, 0..255).
GStreamer runs with the system environment (conda off), as the runbook requires. Read-only: writes only to /tmp.
"""
import argparse
import os
import resource
import subprocess
import tempfile
import time

import cv2
import numpy as np

SYSTEM_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "GST_DEBUG": "2"}


def gst(args, timeout=300):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = time.perf_counter()
    p = subprocess.run(["gst-launch-1.0", "-e", *args], env=SYSTEM_ENV, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    errors = [line for line in (p.stdout + p.stderr).splitlines() if "ERROR" in line or "not-negotiated" in line]
    return wall, cpu, p.returncode, errors[:3]


def boxes_per_path(model, kept, thr=.6):
    """Same model, same frames, colour path changed: box agreement (greedy IoU .5, score >= .6) with today's pixels."""
    import onnxruntime as ort
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    def boxes(bgr):
        x = cv2.dnn.blobFromImage(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), 1 / 255.0, swapRB=True)
        sc, bx = sess.run(["scores", "boxes"], {"images": x, "orig_target_sizes": np.array([[bgr.shape[1], bgr.shape[0]]], np.int64)})
        return bx[0][sc[0] >= thr]

    def agree(a_, b_):
        if not len(a_) and not len(b_):
            return 1.0
        used, tp = set(), 0
        for x in a_:
            best, j = .5, None
            for k, y in enumerate(b_):
                if k in used:
                    continue
                iw = max(0, min(x[2], y[2]) - max(x[0], y[0]))
                ih = max(0, min(x[3], y[3]) - max(x[1], y[1]))
                u = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - iw * ih
                if u > 0 and iw * ih / u >= best:
                    best, j = iw * ih / u, k
            if j is not None:
                used.add(j)
                tp += 1
        return 2 * tp / (len(a_) + len(b_))

    ref = [boxes(f) for f in kept["OpenCV (today)"]]
    print(f"== RT-DETR on today's pixels: {np.mean([len(b) for b in ref]):.1f} boxes per frame (score >= {thr})", flush=True)
    for name, frames in kept.items():
        if name == "OpenCV (today)":
            continue
        ag = [agree(r, boxes(f)) for r, f in zip(ref, frames)]
        print(f"== boxes vs today's pixels, {name}: agreement mean {np.mean(ag):.3f}, worst frame {min(ag):.3f} "
              f"({len(ag)} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--decoder", default="v4l2h264dec")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--model", help="RT-DETR ONNX (images + orig_target_sizes): also compare its boxes per colour path")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{a.video}: {w}x{h}, {n} frames at {fps:.0f} fps ({n / fps:.0f} s) | decoder {a.decoder}", flush=True)

    # 1a. OpenCV CPU decode at the camera pace
    c0, t0 = time.process_time(), time.perf_counter()
    for i in range(n):
        ok, _ = cap.read()
        if not ok:
            break
        time.sleep(max(0.0, t0 + (i + 1) / fps - time.perf_counter()))
    cpu_ocv = (time.process_time() - c0) / (time.perf_counter() - t0)
    print(f"== OpenCV (CPU) decode, paced: {cpu_ocv:.2f} cores busy", flush=True)

    src = ["filesrc", f"location={a.video}", "!", "qtdemux", "!", "h264parse", "!", a.decoder, "!", "video/x-raw,format=NV12"]
    # 1b. hardware decode at the camera pace (fakesink sync=true plays at the file's clock)
    wall, cpu, rc, err = gst(src + ["!", "fakesink", "sync=true"])
    cpu_hw = cpu / wall
    print(f"== hardware decode, paced: {cpu_hw:.2f} cores busy (exit {rc}{', ' + '; '.join(err) if err else ''})", flush=True)
    # 2. hardware decode as fast as it goes
    wall, cpu, rc, err = gst(src + ["!", "fakesink", "sync=false"])
    print(f"== hardware decode, not paced: {n / wall:.0f} frames/s (exit {rc}{', ' + '; '.join(err) if err else ''})", flush=True)

    # 3. hardware decode + GStreamer's own NV12 -> BGR (it knows the decoder's padding), paced: the full cost per stream
    conv = src + ["!", "videoconvert", "!", "video/x-raw,format=BGR"]
    wall, cpu, rc, err = gst(conv + ["!", "fakesink", "sync=true"])
    cpu_conv = cpu / wall
    print(f"== hardware decode + videoconvert to BGR, paced: {cpu_conv:.2f} cores busy (exit {rc}{', ' + '; '.join(err) if err else ''})",
          flush=True)
    # 4. parity on the first frames, for several colour conversions. OpenCV (FFmpeg swscale) converts with BT.601 whatever
    #    the file says; every accuracy number so far was measured on those pixels, so the hardware path must copy them.
    p = subprocess.run(["gst-launch-1.0", "-v", *src, "!", "fakesink", "num-buffers=1"], env=SYSTEM_ENV,
                       capture_output=True, text=True, timeout=60)
    caps = [line.split("caps = ", 1)[1] for line in p.stdout.splitlines() if "src: caps = video/x-raw" in line]
    colorimetry = caps[0].split("colorimetry=(string)", 1)[1].split(",")[0] if caps and "colorimetry=" in caps[0] else "?"
    print(f"== decoder colorimetry: {colorimetry}", flush=True)
    bt601 = ["!", "capssetter", "caps=video/x-raw,colorimetry=bt601"]
    variants = {
        "videoconvert, file colours (BT.709)": src + ["!", "videoconvert"],
        "videoconvert forced BT.601": src + bt601 + ["!", "videoconvert"],
        "videoconvert forced BT.601 + cubic chroma": src + bt601 + ["!", "videoconvert", "chroma-resampler=cubic"],
    }
    cap = cv2.VideoCapture(a.video)
    ref = [cap.read()[1] for _ in range(a.frames)]

    kept = {"OpenCV (today)": ref}

    def compare(name, frames):
        kept[name] = frames
        m = np.array([(float(d.mean()), int(d.max()), float((d > 8).mean()))
                      for d in (np.abs(hb.astype(np.int16) - cb.astype(np.int16)) for hb, cb in zip(frames, ref))])
        print(f"== pixels vs OpenCV, {name}: mean diff {m[:, 0].mean():.2f}, max {int(m[:, 1].max())}, "
              f"share off by > 8: {m[:, 2].mean():.2%} ({len(frames)} frames)", flush=True)

    for name, chain in variants.items():
        raw = os.path.join(tempfile.gettempdir(), "hw_decode_check.bgr")
        _, _, rc, err = gst(chain + ["!", "video/x-raw,format=BGR", "!", "identity", f"eos-after={a.frames}", "!", "filesink",
                                     f"location={raw}"])
        data = np.fromfile(raw, dtype=np.uint8)
        os.remove(raw)
        if rc or data.size % (w * h * 3):
            print(f"== {name}: failed (exit {rc}{', ' + '; '.join(err) if err else ''}, {data.size} bytes)", flush=True)
            continue
        compare(name, list(data.reshape(-1, h, w, 3)))
    # NV12 straight from the decoder (rows padded to a stride), converted by OpenCV in Python
    raw = os.path.join(tempfile.gettempdir(), "hw_decode_check.nv12")
    gst(src + ["!", "identity", f"eos-after={a.frames}", "!", "filesink", f"location={raw}"])
    data = np.fromfile(raw, dtype=np.uint8)
    os.remove(raw)
    stride = next((st for st in range(w, w + 257, 2) if data.size % (st * h * 3 // 2) == 0), None)
    if stride is None:
        print(f"== NV12: {data.size} bytes match no row stride", flush=True)
    else:
        frames, conv = [], []
        for f in data.reshape(-1, h * 3 // 2, stride):
            t = time.process_time()
            frames.append(cv2.cvtColor(np.ascontiguousarray(f[:, :w]), cv2.COLOR_YUV2BGR_NV12))
            conv.append(time.process_time() - t)
        compare(f"NV12 stride {stride} + OpenCV cvtColor", frames)
        print(f"== OpenCV cvtColor NV12->BGR: {np.mean(conv) * 1000:.1f} ms CPU per frame = "
              f"{np.mean(conv) * fps:.2f} cores at {fps:.0f} fps (+ hardware decode {cpu_hw:.2f})", flush=True)
        # FFmpeg swscale, as OpenCV's own video reader converts (7/11 laptop: POINT matched OpenCV to max diff 2)
        try:
            import av
        except ImportError:
            av = None
            print("== PyAV not installed: swscale path skipped", flush=True)
        if av is not None:
            for interp in ("POINT", "BICUBIC"):
                frames, conv = [], []
                for f in data.reshape(-1, h * 3 // 2, stride):
                    t = time.process_time()
                    vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:, :w]), format="nv12")
                    frames.append(vf.reformat(format="bgr24", interpolation=interp).to_ndarray())
                    conv.append(time.process_time() - t)
                compare(f"NV12 + swscale {interp}", frames)
                print(f"== swscale {interp} NV12->BGR: {np.mean(conv) * 1000:.1f} ms CPU per frame = "
                      f"{np.mean(conv) * fps:.2f} cores at {fps:.0f} fps (+ hardware decode {cpu_hw:.2f})", flush=True)
    print(f"== total per stream: OpenCV {cpu_ocv:.2f} cores vs hardware + videoconvert {cpu_conv:.2f} cores", flush=True)
    if a.model:
        boxes_per_path(a.model, kept)

if __name__ == "__main__":
    main()
