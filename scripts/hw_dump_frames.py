"""Direction 2, step 4a (8/11): save hardware-resized 640x640 frames (v4l2h264dec + qtivtransform) for an F1 check.

  python scripts/hw_dump_frames.py OUT_DIR VIDEO.mp4 [VIDEO.mp4 ...] [--every 3 --count 100]

Decodes each video once through the hardware path and keeps frame 0, every, 2*every, ... (count frames): OUT_DIR/<video>.npz
with frames uint8 [N,640,640,3] RGB and their frame indices. The F1 comparison with OpenCV INTER_AREA runs on the laptop
(scripts/eval_preproc_f1.py). Buffer layout measured on 246: packed RGB at the start of a larger buffer.
"""
import argparse
import os
import re
import subprocess
import time

import numpy as np

from media.hw_capture import SYSTEM_ENV, _interrupt_when_parent_dies, gst_source, stop_gracefully

S = 640


def child_cpu(pid):  # user + system seconds of the GStreamer child
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--engine", default="", help="qtivtransform engine: gles (default), fcv, ocv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    caps = f"video/x-raw,format=RGB,width={S},height={S}"
    for video in a.videos:
        chain = gst_source(video, "v4l2h264dec", "auto") + ["!", "qtivtransform", *([f"engine={a.engine}"] if a.engine else []), "!", caps]
        p = subprocess.run(["gst-launch-1.0", "-e", "-v", *chain, "!", "identity", "silent=false", "eos-after=30", "!",
                            "fakesink", "sync=false"], env=SYSTEM_ENV, capture_output=True, text=True, timeout=60,
                           preexec_fn=_interrupt_when_parent_dies)
        log = p.stdout + p.stderr
        sizes = {int(n) for n in re.findall(r"\((\d+) bytes", log)}
        if p.returncode or len(sizes) != 1:
            err = [x.strip() for x in log.splitlines() if "ERROR" in x or "not-negotiated" in x or "erroneous" in x]
            raise SystemExit(f"{video}: hardware pipeline failed (exit {p.returncode}, sizes {sorted(sizes)}): "
                             + " | ".join(err[:3])[:400])
        nbytes = sizes.pop()
        proc = subprocess.Popen(["gst-launch-1.0", "-e", "-q", *chain, "!", "fdsink", "fd=1"], env=SYSTEM_ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                preexec_fn=_interrupt_when_parent_dies)
        buf, frames, idx, k = bytearray(nbytes), [], [], 0
        layouts = set()
        g0, c0 = child_cpu(proc.pid), time.process_time()
        while len(frames) < a.count:
            view, got = memoryview(buf), 0
            while got < nbytes:
                n = proc.stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < nbytes:
                break
            if k % a.every == 0:
                raw = np.frombuffer(buf, np.uint8)
                # 8/11 on 246, same 1638400-byte buffer: engine gles packs 640x640x3 at the start (tail zero),
                # fcv/ocv pad every row to 2560 bytes. shortcut: an all-black bottom quarter would be read as packed.
                packed = not raw[S * S * 3:].any()
                img = raw[:S * S * 3].reshape(S, S, 3) if packed else raw.reshape(S, nbytes // S)[:, :S * 3].reshape(S, S, 3)
                frames.append(img.copy())
                layouts.add("packed" if packed else f"rows padded to {nbytes // S} bytes")
                idx.append(k)
            k += 1
        cpu_ms = (child_cpu(proc.pid) - g0 + time.process_time() - c0) * 1000 / max(k, 1)
        stop_gracefully(proc)
        name = os.path.splitext(os.path.basename(video))[0]
        np.savez_compressed(os.path.join(a.out, f"{name}.npz"), frames=np.stack(frames), index=np.array(idx))
        print(f"{name}: {len(frames)} frames (every {a.every}) -> {a.out}/{name}.npz | engine {a.engine or 'default'}: "
              f"decode+transform+read {cpu_ms:.1f} CPU-ms per frame = {cpu_ms * 15 / 1000:.2f} cores at 15 fps | "
              f"layout {sorted(layouts)}", flush=True)


if __name__ == "__main__":
    main()
