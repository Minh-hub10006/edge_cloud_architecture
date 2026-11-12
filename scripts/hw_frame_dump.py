"""What does the model really see? (13/11) Saves the hardware decoder's 640x640 frames next to OpenCV's frames.

  python -B scripts/hw_frame_dump.py VIDEO.mp4 --decoder qtic2vdec --out ~/hw_dump [--frames 0 300 1000]

For each index: <out>/hw_<i>.jpg (HwCapture, what the model gets), <out>/cpu_<i>.jpg (OpenCV, stretched to 640x640
the same way) and the mean pixel difference (same picture: a few levels; a sheared / shifted picture: tens).
"""
import argparse
import os

import cv2
import numpy as np

from media.hw_capture import HwCapture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--decoder", default="qtic2vdec")
    ap.add_argument("--out", default=os.path.expanduser("~/hw_dump"))
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 300, 1000])
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
    hw.release()


if __name__ == "__main__":
    main()
