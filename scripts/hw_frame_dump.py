"""What does the model really see? (13/11) Saves the hardware decoder's 640x640 frames next to OpenCV's frames.

  python -B scripts/hw_frame_dump.py VIDEO.mp4 --decoder qtic2vdec --out ~/hw_dump [--frames 0 300 1000]

For each index: <out>/hw_<i>.jpg (HwCapture, what the model gets), <out>/cpu_<i>.jpg (OpenCV, stretched to 640x640
the same way) and the mean pixel difference (same picture: a few levels; a sheared / shifted picture: tens).
"""
import argparse
import os
import time

import cv2
import numpy as np

from media.hw_capture import HwCapture, stop_gracefully


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--decoder", default="qtic2vdec")
    ap.add_argument("--out", default=os.path.expanduser("~/hw_dump"))
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 300, 1000])
    ap.add_argument("--seek", type=int, nargs="*", default=[], help="then jump to these frames (restart path, 14/11)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hw, cpu = HwCapture(a.video, decoder=a.decoder, size=640, hires_every=5), cv2.VideoCapture(a.video)
    print(f"buffer bytes per frame {hw.frame_bytes} (640x640x3 = {640 * 640 * 3}, extra {hw.frame_bytes - 640 * 640 * 3})")
    want, i = set(a.frames) | {0}, 0
    while want:
        ok, px = hw.read()
        ok2, ref = cpu.read()
        if not (ok and ok2):
            print(f"stopped at frame {i}: hw {ok} cpu {ok2}")
            break
        if i == 0:   # 13/11: on 252 this began with 64 KiB of decoder log text (fixed: frames have their own pipe)
            open(f"{a.out}/raw_0.bin", "wb").write(bytes(hw._buf))
        if i in want:
            want.discard(i)
            ref = cv2.resize(ref, (640, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{a.out}/hw_{i}.jpg", px)
            cv2.imwrite(f"{a.out}/cpu_{i}.jpg", ref)
            diff = float(np.abs(px.astype(int) - ref.astype(int)).mean())
            print(f"frame {i}: hw {px.shape} vs cpu, mean difference {diff:.1f} levels "
                  f"-> {'same picture' if diff < 12 else 'DIFFERENT picture'}", flush=True)
        i += 1
    for target in a.seek:   # HwCapture.set = what a decoder restart does: from the keyframe before, never from frame 0
        t = time.monotonic()
        ok = hw.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, px = hw.read() if ok else (False, None)
        took = time.monotonic() - t
        cpu.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok2, ref = cpu.read()
        if not (ok and ok2):
            print(f"seek {target}: hw {ok} cpu {ok2} -> FAILED")
            continue
        diff = float(np.abs(px.astype(int) - cv2.resize(ref, (640, 640), interpolation=cv2.INTER_AREA).astype(int)).mean())
        print(f"seek {target}: from keyframe {hw.keyframe_before(target)[0]} in {took:.1f} s, mean difference {diff:.1f} "
              f"-> {'same picture' if diff < 12 else 'DIFFERENT picture'}", flush=True)
        for i in range(target + 1, target + 12):   # the full-resolution branch after the seek: same frame as the main one
            ok, _ = hw.read()
            ok2, ref = cpu.read()
            if ok and ok2 and hw.last_hires is not None:
                d = float(np.abs(hw.last_hires.astype(int) - ref.astype(int)).mean())
                print(f"   full frame at {i}: mean difference {d:.1f} -> {'same picture' if d < 12 else 'DIFFERENT picture'}")
                break
        else:
            print("   no full-resolution frame within 11 frames after the seek -> FAILED")
    # never forced: 13/11 a forced stop on 246 left a ghost decoder session (CMA 16 MB, every camera failed)
    if not stop_gracefully(hw._proc, timeout_s=30, force=False):
        print("!! the decoder did not stop in 30 s: left running, it ends at the end of the file")
    if hw._feeder is not None:
        hw._feeder.kill()


if __name__ == "__main__":
    main()
