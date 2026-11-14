"""Which hardware decode branch faults the GPU on this Edge? (14/11: 252 logged "kgsl GPU PAGE FAULT ... gst-launch-1.0"
in every multi-stream run, then decoder error -5 and dead cameras.)

  python3 -B scripts/hw_stress.py VIDEO.mp4 --decoder qtic2vdec [--streams 4] [--seconds 60]   (as root: reads dmesg)

Runs N hardware captures at once at 15 fps, first WITHOUT the full-resolution branch (640x640 only), then WITH it
(every 5th frame at full size), each for --seconds. Per phase: frames per stream, decoder stalls / restarts / dead,
and the GPU page faults the kernel logged meanwhile. No model, no Cloud, nothing written.
"""
import argparse
import subprocess
import threading
import time

from media.hw_capture import HwCapture


def gpu_faults():
    out = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    return out.count("GPU PAGE FAULT"), out.count("msm_vidc:   err")


def phase(video, decoder, streams, seconds, hires_every):
    caps = [HwCapture(video, decoder=decoder, size=640, hires_every=hires_every) for _ in range(streams)]
    frames, hires = [0] * streams, [0] * streams
    end = time.monotonic() + seconds

    def run(i):
        t0, n = time.monotonic(), 0
        while time.monotonic() < end:
            ok, _ = caps[i].read()
            if not ok:
                return
            frames[i] += 1
            hires[i] += caps[i].last_hires is not None
            n += 1
            time.sleep(max(0.0, t0 + n / 15 - time.monotonic()))
    f0, e0 = gpu_faults()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    f1, e1 = gpu_faults()
    for c in caps:
        c.release()
    print(f"full-res every {hires_every or '-'}: frames per stream {frames} (expected ~{seconds * 15}), full frames "
          f"{hires}, stalls {[c.stalls for c in caps]}, restarts {[c.restarts for c in caps]}, "
          f"dead {[c.dead for c in caps]} | GPU page faults +{f1 - f0}, msm_vidc errors +{e1 - e0}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--decoder", default="qtic2vdec")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    phase(a.video, a.decoder, a.streams, a.seconds, 0)
    time.sleep(5)
    phase(a.video, a.decoder, a.streams, a.seconds, 5)


if __name__ == "__main__":
    main()
