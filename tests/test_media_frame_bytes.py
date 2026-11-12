"""Single-copy media frame bytes are identical to the old resize+concatenate+tobytes path."""
import unittest

import numpy as np

from edge_agent import CapturedFrame
from contracts import FrameIdentity
from media.ffmpeg_publisher import _frame_bytes
from media.frame_identity import append_identity, read_identity
from media.profile import MediaProfile

KEY = ("edge-1", "cam", "stream", 7)


def old_bytes(pixels, profile, identity):
    import cv2
    if pixels.shape[:2] != (profile.height, profile.width):
        shrinking = pixels.shape[0] > profile.height or pixels.shape[1] > profile.width
        pixels = cv2.resize(pixels, (profile.width, profile.height),
                            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    if identity is not None:
        pixels = append_identity(pixels, **identity)
    return np.ascontiguousarray(pixels).tobytes()


def frame(pixels):
    return CapturedFrame(FrameIdentity(*KEY, 0, 1, 0), pixels, 0)


class FrameBytesTests(unittest.TestCase):
    def test_identical_bytes_old_vs_single_copy(self):
        rng = np.random.default_rng(0)
        profile = MediaProfile(1280, 720, 15, 2500, 15)
        for shape in ((1080, 1920, 3), (720, 1280, 3), (480, 640, 3)):
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            for identity in (None, dict(key=KEY, seq=41, fps=15), dict(key=KEY, seq=42, fps=15, padding=True)):
                stages = {}
                new = _frame_bytes(frame(pixels), profile, identity, stages)
                self.assertEqual(new, old_bytes(pixels, profile, identity))
                self.assertIn("media_resize_ms", stages)

    def test_band_still_decodes(self):
        profile = MediaProfile(1280, 720, 15, 2500, 15)
        raw = _frame_bytes(frame(np.zeros((1080, 1920, 3), np.uint8)), profile, dict(key=KEY, seq=9, fps=15))
        decoded = read_identity(np.frombuffer(raw, np.uint8).reshape(784, 1280, 3), key=KEY)
        self.assertEqual((decoded.seq, decoded.fps, decoded.padding), (9, 15, False))


class OverlappedWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_previous_frame_drained_before_next_write_and_on_close(self):
        from media.ffmpeg_publisher import FFmpegH264Publisher
        calls = []

        class Stdin:
            def write(self, data):
                calls.append(("write", data))
            async def drain(self):
                calls.append(("drain",))
            def close(self):
                calls.append(("close",))
            async def wait_closed(self):
                pass

        class Process:
            stdin, returncode = Stdin(), None
            async def wait(self):
                return 0

        publisher = FFmpegH264Publisher(profile=MediaProfile(1280, 720, 15, 2500, 15),
                                        output="rtsp://x/y", stream_key=KEY)
        publisher._process = Process()
        pixels = np.zeros((720, 1280, 3), np.uint8)
        for seq in range(3):
            await publisher.publish(frame(pixels + seq))
        await publisher.close()
        kinds = [c[0] for c in calls]
        # drain(previous) -> write(n), three times, then drain(last) before close.
        self.assertEqual(kinds, ["drain", "write"] * 3 + ["drain", "close"])
        self.assertEqual([c[1][0] for c in calls if c[0] == "write"], [0, 1, 2])  # order kept


if __name__ == "__main__":
    unittest.main()
