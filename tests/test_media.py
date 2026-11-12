"""Local media tests require no RTSP server or downloaded dependency."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from contracts import FrameIdentity
from edge_agent import CapturedFrame, EdgeFanout, SyntheticFrameSource
from media import (
    FFmpegH264Publisher, MediaCadence, MediaProfile, OpenCVFrameSource,
    build_ffmpeg_command, media_path, rtsp_publish_url, run_media_pipeline,
    webrtc_page_url,
)


KEY = ("edge-01", "cam-01", "main", 3)


class MediaContractTests(unittest.TestCase):
    def test_stream_path_and_urls(self) -> None:
        path = "edge/edge-01/cam-01/main/epoch-3"
        self.assertEqual(media_path(KEY), path)
        self.assertEqual(rtsp_publish_url("rtsp://localhost:8554", KEY), "rtsp://localhost:8554/" + path)
        self.assertEqual(webrtc_page_url("http://localhost:8889", KEY), "http://localhost:8889/" + path)
        with self.assertRaises(ValueError):
            media_path(("other/node", "cam", "main", 0))
        with self.assertRaises(ValueError):
            rtsp_publish_url("http://localhost:8554", KEY)

    def test_low_latency_encoder_settings(self) -> None:
        profile = MediaProfile(width=64, height=48, fps=15, bitrate_kbps=300, gop_frames=15)
        command = build_ffmpeg_command(profile, "rtsp://localhost:8554/camera")
        self.assertIn("zerolatency", command)
        self.assertEqual(command[command.index("-bf") + 1], "0")
        self.assertEqual(command[command.index("-trellis") + 1], "0")  # same x264 settings on every Edge
        self.assertEqual(command[command.index("-rtsp_transport") + 1], "tcp")
        self.assertEqual(command[-3:], ["-f", "rtsp", "rtsp://localhost:8554/camera"])
        with self.assertRaises(ValueError):
            MediaProfile(width=63)

    def test_media_cadence(self) -> None:
        cadence = MediaCadence(5)
        self.assertEqual(
            [cadence.accept(t) for t in (0, 100_000_000, 200_000_000, 300_000_000, 400_000_000)],
            [True, False, True, False, True],
        )
        with self.assertRaises(ValueError):
            cadence.accept(399_000_000)

    def test_cadence_accepts_decode_jitter_without_dropping_half_the_video(self) -> None:
        cadence = MediaCadence(30)
        accepted = [cadence.accept(round(i * 1e9 / 30) + (3_000_000 if i % 2 == 0 else 0))
                    for i in range(900)]
        self.assertEqual(sum(accepted), 900)

    def test_cadence_preserves_downsample_rate_and_rejects_bursts(self) -> None:
        for source_fps, target_fps in ((30, 15), (60, 30), (30, 5)):
            cadence = MediaCadence(target_fps)
            self.assertEqual(sum(cadence.accept(round(i * 1e9 / source_fps))
                                 for i in range(source_fps * 10)), target_fps * 10)
        cadence = MediaCadence(30)
        self.assertTrue(cadence.accept(0))
        self.assertFalse(cadence.accept(1_000_000))
        self.assertTrue(cadence.accept(1_000_000_000))
        self.assertFalse(cadence.accept(1_001_000_000))

    def test_selection_clock_is_separate_from_latency_clock(self) -> None:
        identity = FrameIdentity(node_id=KEY[0], camera_id=KEY[1], stream_id=KEY[2],
                                 stream_epoch=KEY[3], source_frame_id=0, capture_utc_ns=1, source_pts=0)
        frame = CapturedFrame(identity, None, 123_000_000, cadence_ns=0)
        self.assertEqual(frame.cadence_time_ns, 0)
        self.assertEqual(frame.capture_mono_ns, 123_000_000)
        self.assertEqual(CapturedFrame(identity, None, 123).cadence_time_ns, 123)
        with self.assertRaises(ValueError):
            CapturedFrame(identity, None, 123, cadence_ns=-1)


class MediaPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_fanout_and_preview_cadence(self) -> None:
        class CollectingPublisher:
            profile = MediaProfile(width=64, height=48, fps=5)

            def __init__(self) -> None:
                self.frames_submitted = 0
                self.ids: list[int] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args) -> None:
                return None

            async def publish(self, frame: CapturedFrame) -> None:
                self.frames_submitted += 1
                self.ids.append(frame.identity.source_frame_id)

        source = SyntheticFrameSource(
            node_id=KEY[0], camera_id=KEY[1], stream_id=KEY[2],
            stream_epoch=KEY[3], frame_count=10, start_utc_ns=1,
            start_mono_ns=0, frame_interval_ns=100_000_000,
        )
        fanout = EdgeFanout(media_capacity=10, analytics_capacity=10)
        publisher = CollectingPublisher()
        stats = await run_media_pipeline(source, fanout, publisher)
        self.assertEqual(stats.captured, 10)
        self.assertEqual(stats.media_queue_dropped, 0)
        self.assertEqual(stats.encoded_frames, 5)
        self.assertEqual(stats.preview_fps_skipped, 5)
        self.assertEqual(publisher.ids, [0, 2, 4, 6, 8])
        self.assertEqual(fanout.analytics.pending, 10)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    async def test_real_h264_encode_and_decode(self) -> None:
        try:
            import numpy as np
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("numpy/OpenCV unavailable")

        profile = MediaProfile(width=64, height=48, fps=15, bitrate_kbps=300, gop_frames=5)
        references = []

        async def collect(reference) -> None:
            references.append(reference)

        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "preview.h264")
            async with FFmpegH264Publisher(
                profile=profile, output=output, stream_key=KEY, output_format="h264",
                on_frame_submitted=collect,
            ) as publisher:
                for i in range(12):
                    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
                    pixels[:, :, i % 3] = 30 + i * 10
                    frame = CapturedFrame(
                        identity=FrameIdentity(
                            node_id=KEY[0], camera_id=KEY[1], stream_id=KEY[2],
                            stream_epoch=KEY[3], source_frame_id=i,
                            capture_utc_ns=1 + i, source_pts=i,
                        ),
                        pixels=pixels,
                        capture_mono_ns=i,
                    )
                    await publisher.publish(frame)
                for _ in range(3):
                    await publisher.publish_padding(frame)
            self.assertEqual(publisher.frames_submitted, 12)
            self.assertEqual(publisher.padding_frames_submitted, 3)
            self.assertEqual([reference.encoded_seq for reference in references], list(range(12)))
            self.assertEqual([reference.media_pts_ns for reference in references[:3]], [0, 66_666_667, 133_333_333])
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,nb_read_frames",
                    "-of", "json", output,
                ],
                capture_output=True, text=True, check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual((stream["width"], stream["height"]), (64, 48))
            self.assertEqual(int(stream["nb_read_frames"]), 15)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    async def test_opencv_file_source_decodes_once(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            video = str(Path(tmp) / "source.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x48:rate=10", "-frames:v", "5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", video,
                ],
                capture_output=True, check=True,
            )
            source = OpenCVFrameSource(
                video, node_id=KEY[0], camera_id=KEY[1],
                stream_id=KEY[2], stream_epoch=KEY[3],
                realtime_file=False,
            )
            frames = [frame async for frame in source]
            self.assertEqual([frame.identity.source_frame_id for frame in frames], list(range(5)))
            self.assertTrue(all(frame.pixels.shape == (48, 64, 3) for frame in frames))
            self.assertEqual(len({frame.identity.source_pts for frame in frames}), 5)
            self.assertEqual([frame.cadence_time_ns for frame in frames],
                             [i * 100_000_000 for i in range(5)])
            self.assertEqual(frames[0].identity.source_pts, 0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
    async def test_file_to_fanout_to_h264_end_to_end(self) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")

        profile = MediaProfile(width=64, height=48, fps=5, bitrate_kbps=300, gop_frames=5)
        with tempfile.TemporaryDirectory() as tmp:
            video = str(Path(tmp) / "source.mp4")
            output = str(Path(tmp) / "preview.h264")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x48:rate=10", "-frames:v", "10",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", video,
                ],
                capture_output=True, check=True,
            )
            source = OpenCVFrameSource(
                video, node_id=KEY[0], camera_id=KEY[1],
                stream_id=KEY[2], stream_epoch=KEY[3],
            )
            fanout = EdgeFanout(media_capacity=8, analytics_capacity=10)
            publisher = FFmpegH264Publisher(
                profile=profile, output=output, stream_key=KEY, output_format="h264",
            )
            stats = await run_media_pipeline(source, fanout, publisher)
            self.assertEqual(stats.captured, 10)
            self.assertEqual(stats.media_queue_dropped, 0)
            self.assertEqual(stats.analytics_queue_dropped, 0)
            self.assertGreaterEqual(stats.encoded_frames, 4)
            self.assertLessEqual(stats.encoded_frames, 6)
            self.assertEqual(stats.encoded_frames + stats.preview_fps_skipped, 10)
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,nb_read_frames",
                    "-of", "json", output,
                ],
                capture_output=True, text=True, check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(int(stream["nb_read_frames"]), stats.encoded_frames)


if __name__ == "__main__":
    unittest.main()
