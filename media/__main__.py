"""Publish one video file through the shared Edge capture path."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json

from edge_agent import EdgeFanout

from .ffmpeg_publisher import FFmpegH264Publisher
from .opencv_source import OpenCVFrameSource
from .pipeline import run_media_pipeline
from .profile import MediaProfile, rtsp_publish_url, webrtc_page_url


async def _run(args: argparse.Namespace) -> None:
    profile = MediaProfile(
        width=args.width, height=args.height, fps=args.preview_fps,
        bitrate_kbps=args.bitrate_kbps, gop_frames=args.preview_fps,
    )
    key = (args.node_id, args.camera_id, args.stream_id, args.stream_epoch)
    source = OpenCVFrameSource(
        args.video, node_id=args.node_id, camera_id=args.camera_id,
        stream_id=args.stream_id, stream_epoch=args.stream_epoch,
        max_frames=args.frames,
    )
    fanout = EdgeFanout(media_capacity=args.media_queue, analytics_capacity=args.analytics_queue)
    publisher = FFmpegH264Publisher(
        profile=profile, stream_key=key,
        output=rtsp_publish_url(args.rtsp_base, key),
    )
    stats = await run_media_pipeline(source, fanout, publisher)
    print(json.dumps({
        **asdict(stats),
        "rtsp_url": publisher.output,
        "webrtc_url": webrtc_page_url(args.webrtc_base, key),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 H.264/RTSP media POC")
    parser.add_argument("--video", required=True, help="Video file decoded once at Edge")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--node-id", default="edge-01")
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument("--stream-id", default="main")
    parser.add_argument("--stream-epoch", type=int, default=0)
    parser.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--webrtc-base", default="http://127.0.0.1:8889")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--bitrate-kbps", type=int, default=800)
    parser.add_argument("--media-queue", type=int, default=8)
    parser.add_argument("--analytics-queue", type=int, default=8)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
