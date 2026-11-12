"""Prepare a testbed main stream (6/11): H.264 without B-frames, with the frame number in the identity band.

  python -B scripts/make_main_video.py SOURCE.mp4 OUT.mp4 --frames 2100 [--bitrate 4M]

The band (media/frame_identity.py, 64 rows under the picture) lets a reader get the exact frame from the picture itself.
The source loops as many times as needed and the number keeps growing, as the Edge's frame numbers do: frame n of OUT
shows the same picture as analytics frame n of a run started at frame 0. Encoded once here, then the Edge only forwards.
"""
import argparse
import subprocess
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media.copy_publisher import MAIN_KEY  # noqa: E402
from media.frame_identity import append_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--bitrate", default="4M")
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"Refusing to overwrite {a.out}")
    cap = cv2.VideoCapture(a.source)
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), round(cap.get(cv2.CAP_PROP_FPS))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{w}x{h + 64}",
         "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-profile:v", "main", "-bf", "0",
         "-b:v", a.bitrate, "-maxrate", a.bitrate, "-bufsize", a.bitrate, "-g", str(fps), "-pix_fmt", "yuv420p", a.out],
        stdin=subprocess.PIPE)
    for seq in range(a.frames):
        ok, pixels = cap.read()
        if not ok:  # same rewind as media/opencv_source.py
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, pixels = cap.read()
        assert ok, "cannot read the source"
        ffmpeg.stdin.write(append_identity(pixels, key=MAIN_KEY, seq=seq, fps=fps).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise SystemExit("ffmpeg failed")
    print(f"{a.out}: {a.frames} frames, {w}x{h}+64 band, {fps} fps")


if __name__ == "__main__":
    main()
