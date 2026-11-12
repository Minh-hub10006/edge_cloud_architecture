"""Re-encode videos the way an Edge publishes them to Cloud, so Cloud-model quality is scored on
what Cloud actually sees: same resize, identity band and x264 flags (media/ffmpeg_publisher.py).

  python -B scripts/live_transcode.py IN_DIR OUT_DIR [--live-fps 15 --live-kbps 2500]

Every source frame is kept, so frame numbers still match the annotations; the bitrate is scaled by
source_fps / live_fps and CALIBRATION so picture quality (PSNR) matches live, and the GOP stays 1 s as live.
Dump the result with `edge_parity.py dump --crop-bottom 64` (QUALITY_CROP_BOTTOM=64), as Cloud crops the band.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media.ffmpeg_publisher import _frame_bytes, build_ffmpeg_command  # noqa: E402
from media.profile import MediaProfile  # noqa: E402

KEY = ("edge-sim", "detrac", "live", 0)
# 25 fps encodes more efficiently than live 15 fps (less motion per frame), so bits/frame alone overshoots
# quality by 0.5-1.2 dB; 0.76 matched the live-exact encode's mean PSNR within ~0.3 dB on night/sunny/rainy clips.
CALIBRATION = 0.76


def transcode(src, dst, *, width=1280, height=720, live_fps=15, live_kbps=2500, ffmpeg="ffmpeg") -> int:
    import cv2
    src, dst = Path(src), Path(dst)
    capture = cv2.VideoCapture(str(src))
    fps = round(capture.get(cv2.CAP_PROP_FPS)) or 25
    profile = MediaProfile(width=width, height=height, fps=fps,
                           bitrate_kbps=round(live_kbps * CALIBRATION * fps / live_fps), gop_frames=fps)
    raw, part = dst.with_suffix(".h264.part"), dst.with_suffix(".tmp.mp4")
    encoder = subprocess.Popen(build_ffmpeg_command(profile, str(raw), output_format="h264", ffmpeg_bin=ffmpeg,
                                                    frame_identity=True), stdin=subprocess.PIPE)
    frames = 0
    try:
        while True:
            ok, pixels = capture.read()
            if not ok:
                break
            encoder.stdin.write(_frame_bytes(SimpleNamespace(pixels=pixels), profile,
                                             identity={"key": KEY, "seq": frames, "fps": fps}))
            frames += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frames == 0:
        raise RuntimeError(f"encode failed for {src} after {frames} frames")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(raw),
                    "-c", "copy", str(part)], check=True)
    raw.unlink()
    part.replace(dst)
    return frames


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_dir")
    p.add_argument("out_dir")
    p.add_argument("--live-fps", type=int, default=15)
    p.add_argument("--live-kbps", type=int, default=2500)
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.in_dir).glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 in {args.in_dir}")
    for i, video in enumerate(videos, 1):
        dst = out_dir / video.name
        if dst.exists():  # resumable
            continue
        frames = transcode(video, dst, live_fps=args.live_fps, live_kbps=args.live_kbps, ffmpeg=args.ffmpeg)
        print(f"[LT] {i}/{len(videos)} {video.name} frames={frames}", flush=True)
    print(f"[LT] done {len(videos)} videos -> {out_dir}")


if __name__ == "__main__":
    main()
