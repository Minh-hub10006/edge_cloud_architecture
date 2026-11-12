"""Which way of ending a hardware decode leaves a ghost decoder session? (8/11, 246: after the frame dumps msm_vidc listed
6 sessions "1152 | 1920" with no process left, CmaFree 125 -> 14 MB, ~14 MB each, idle CmaFree flat -> our runs leak.)

  python scripts/hw_leak_steps.py VIDEO.mp4        (right after a reboot)

Each step runs twice; CmaFree is read 3 s after each run. A drop of ~10+ MB = a ghost session. Stops when CmaFree < 40 MB.
  a  decode only,        ended by EOS (identity eos-after=30)        = the probe HwCapture runs first
  b  decode only,        HwCapture(size=0): probe + read 60 + SIGINT mid-video
  c  decode + qtivtransform gles, ended by EOS
  d  decode + qtivtransform gles, HwCapture(size=640): probe + read 60 + SIGINT mid-video
  e  decode + qtivtransform gles, read to the end of the video (no stop signal)
"""
import subprocess
import sys
import time

from media.hw_capture import SYSTEM_ENV, HwCapture, _interrupt_when_parent_dies, gst_source, transform_chain


def cma():
    return next(int(x.split()[1]) // 1024 for x in open("/proc/meminfo") if x.startswith("CmaFree"))


def eos_run(video, chain_tail):
    p = subprocess.run(["gst-launch-1.0", "-e", *gst_source(video, "v4l2h264dec"), *chain_tail, "!", "identity",
                        "eos-after=30", "!", "fakesink", "sync=false"], env=SYSTEM_ENV, capture_output=True, text=True,
                       timeout=120, preexec_fn=_interrupt_when_parent_dies)
    return f"exit {p.returncode}"


def capture_run(video, size, frames=60):
    cap = HwCapture(video, size=size)
    n = 0
    while (frames is None or n < frames) and cap.read()[0]:
        n += 1
    cap.release()
    return f"read {n} frames"


def main():
    video = sys.argv[1]
    steps = [("a decode only, EOS", lambda: eos_run(video, [])),
             ("b decode only, SIGINT mid-video", lambda: capture_run(video, 0)),
             ("c decode+gles, EOS", lambda: eos_run(video, transform_chain(640))),
             ("d decode+gles, SIGINT mid-video", lambda: capture_run(video, 640)),
             ("e decode+gles, to end of video", lambda: capture_run(video, 640, frames=None))]
    print(f"start: CmaFree {cma()} MB", flush=True)
    for name, run in steps:
        for i in (1, 2):
            before = cma()
            if before < 40:
                print("CmaFree < 40 MB -> stop (reboot to free it)", flush=True)
                return
            try:
                what = run()
            except Exception as exc:  # report and stop: a failed open may itself leak
                print(f"{name} #{i}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                return
            time.sleep(3)
            after = cma()
            print(f"{name} #{i}: {what}, CmaFree {before} -> {after} MB ({after - before:+d}){'  <- LEAK' if before - after >= 8 else ''}",
                  flush=True)


if __name__ == "__main__":
    main()
