"""Does stopping a hardware reader mid-video leave the decoder a ghost session? (8/11: msm_vidc once kept 1080p sessions
of finished runs and refused new ones with ENOMEM until a reboot.)

  python scripts/hw_restart_check.py VIDEO.mp4 [--rounds 10 --size 640]

Each round opens media.hw_capture.HwCapture exactly as the Edge Agent does, reads 60 frames, stops it mid-video. Prints
CmaFree and the number of stale-session lines msm_vidc wrote to dmesg (needs sudo for dmesg; -1 if not readable).
"""
import argparse
import subprocess

from media.hw_capture import HwCapture


def cma_mb():
    return next(int(x.split()[1]) // 1024 for x in open("/proc/meminfo") if x.startswith("CmaFree"))


def stale_lines():
    p = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True)
    return p.stdout.count("avcD_0:   1088") if p.returncode == 0 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    print(f"before: CmaFree {cma_mb()} MB, stale-session lines {stale_lines()}", flush=True)
    for r in range(1, a.rounds + 1):
        cap = HwCapture(a.video, size=a.size)
        n = sum(1 for _ in range(60) if cap.read()[0])
        cap.release()
        print(f"round {r:2d}: read {n}/60 frames {cap.size}x{cap.size}, CmaFree {cma_mb()} MB, "
              f"stale-session lines {stale_lines()}", flush=True)
        if n < 60:
            print("decoder stopped early -> stop here; " + cap.error_tail[-300:], flush=True)
            break


if __name__ == "__main__":
    main()
