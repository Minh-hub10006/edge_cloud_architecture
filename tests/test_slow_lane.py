"""Slow lane step 1 (8/11): BUS, frame ring, full-resolution frames from the hardware branch and the CPU path."""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np


def frame(i, ms, hires=None):
    return SimpleNamespace(identity=SimpleNamespace(source_frame_id=i), capture_mono_ns=ms * 1_000_000,
                           pixels=np.zeros((4, 4, 3), np.uint8), hires=hires)


class BusAndRingTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_task_drops_its_own_oldest_events_and_never_blocks_the_publisher(self):
        from edge_agent.slow_lane import MetadataBus, SlowEvent
        bus = MetadataBus()
        slow, fast = bus.subscribe("slow", 2), bus.subscribe("fast", 8)
        for i in range(5):
            bus.publish(SlowEvent(frame(i, i), None))   # returns at once, nothing is awaited
        self.assertEqual((slow.dropped, fast.dropped), (3, 0))
        self.assertEqual((await slow.get()).frame.identity.source_frame_id, 3)  # the newest two are kept
        with self.assertRaises(ValueError):
            bus.subscribe("slow")

    def test_ring_keeps_seconds_finds_hires_and_spaced_sequences(self):
        from edge_agent.slow_lane import FrameRing
        ring = FrameRing(seconds=1.0, hires_seconds=.5)
        big = np.ones((8, 8, 3), np.uint8)
        for i in range(30):   # 15 fps for 2 s, full resolution on every 3rd frame
            ring.add(frame(i, round(i * 1000 / 15), big if i % 3 == 0 else None))
        ids = [f.identity.source_frame_id for f in ring.frames]
        self.assertEqual((ids[0], ids[-1]), (14, 29))                        # only the last second
        self.assertEqual(ring.hires_near(28).identity.source_frame_id, 27)
        self.assertIsNone(ring.hires_near(5))                                 # already left the ring
        self.assertIsNone(ring.hires_near(17))                                # full frames kept only 0.5 s
        self.assertEqual([f.identity.source_frame_id for f in ring.hires], [24, 27])
        seq = ring.sequence(4, every_ns=200_000_000)                          # 5 per s
        self.assertEqual([f.identity.source_frame_id for f in seq], [20, 23, 26, 29])
        gaps = np.diff([f.capture_mono_ns for f in seq])
        self.assertEqual(len(seq), 4)
        self.assertTrue((gaps >= 200_000_000).all() and seq[-1].identity.source_frame_id == 29)

    async def test_lane_runs_tasks_until_the_media_lane_stops_and_reports_them(self):
        from edge_agent.slow_lane import build_slow_lane
        lane = build_slow_lane("check")
        stop = asyncio.Event()
        run = asyncio.create_task(lane.run(stop))
        big = np.ones((8, 8, 3), np.uint8)
        for i in range(6):
            f = frame(i, i * 66, big if i % 3 == 0 else None)
            lane.on_frame(f)
            lane.on_result(f, None)
            await asyncio.sleep(0)
        stop.set()
        stats = await asyncio.wait_for(run, 2)
        self.assertEqual(stats["published"], 6)
        self.assertEqual(stats["tasks"]["check"]["events"], 6)
        self.assertEqual(stats["tasks"]["check"]["with_hires"], 6)           # every frame within 2 of a full one
        self.assertEqual(stats["tasks"]["check"]["hires_shape"], [8, 8, 3])
        with self.assertRaises(ValueError):
            build_slow_lane("check,nope")
        self.assertIsNone(build_slow_lane(""))


class HiresBranchTests(unittest.TestCase):
    def test_buffers_of_any_size_in_both_layouts(self):
        from media.hw_capture import bgr_from_buffer
        w, h = 6, 4
        img = np.random.default_rng(2).integers(1, 255, (h, w, 3), np.uint8)
        packed = np.zeros(h * 32, np.uint8)
        packed[:w * h * 3] = img.ravel()
        padded = np.zeros((h, 32), np.uint8)
        padded[:, :w * 3] = img.reshape(h, -1)
        for buf in (packed.tobytes(), padded.tobytes()):
            self.assertTrue((bgr_from_buffer(buf, w, h) == img).all())

    def test_full_resolution_frame_k_belongs_to_analytics_frame_k_times_n(self):
        from media.hw_capture import HwCapture
        cap = object.__new__(HwCapture)
        cap.width, cap.height, cap.hires_every = 2, 2, 3
        cap.hires_bytes = 2 * 2 * 3
        cap._hires, cap._hires_cv = {}, threading.Condition()
        r, w = os.pipe()
        reader = threading.Thread(target=cap._read_hires, args=(os.fdopen(r, "rb", buffering=0), cap._hires,
                                                                cap._hires_cv))
        reader.start()
        for k in range(3):
            os.write(w, bytes([k + 1]) * cap.hires_bytes)
        os.close(w)
        reader.join(2)
        self.assertEqual(cap._hires_for(3)[0, 0, 0], 2)     # analytics frame 3 = branch frame 1
        self.assertIsNone(cap._hires_for(4))                # not a multiple of 3: no full frame
        self.assertEqual(cap._hires_for(6)[0, 0, 0], 3)
        self.assertIsNone(cap._hires_for(9, timeout_s=.2))  # branch ended: no wait


class CpuPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_cpu_frames_carry_themselves_as_full_resolution_on_every_nth_frame(self):
        import cv2
        from media.opencv_source import OpenCVFrameSource
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.avi")
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15, (32, 24))
            for i in range(7):
                out.write(np.full((24, 32, 3), i * 30, np.uint8))
            out.release()
            src = OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s", realtime_file=False, hires_every=3)
            frames = [f async for f in src]
        self.assertEqual([f.hires is f.pixels for f in frames], [i % 3 == 0 for i in range(7)])
        self.assertTrue(all(f.hires is None for f in frames if f.identity.source_frame_id % 3))


if __name__ == "__main__":
    unittest.main()


class RingMemoryTests(unittest.TestCase):
    def test_the_long_ring_does_not_pin_full_resolution_arrays(self):
        from contracts import FrameIdentity
        from edge_agent.capture import CapturedFrame
        from edge_agent.slow_lane import FrameRing
        ring = FrameRing(seconds=3.0, hires_seconds=1.0)
        big = np.ones((100, 100, 3), np.uint8)
        for i in range(30):
            ident = FrameIdentity("n", "c", "s", 0, i, 1, i)
            ring.add(CapturedFrame(ident, np.zeros((10, 10, 3), np.uint8), i * 200_000_000, hires=big if i % 5 == 0 else None))
        self.assertTrue(all(f.hires is None for f in ring.frames))           # 3 s of small frames only
        self.assertEqual([f.identity.source_frame_id for f in ring.hires], [25])   # 1 s of full frames
        self.assertAlmostEqual(ring.megabytes, (len(ring.frames) * 300 + 30000) / 1e6)


class ProbeStopTests(unittest.TestCase):
    def test_a_hung_decoder_probe_is_stopped_with_sigint_not_killed(self):
        import signal
        import subprocess
        from unittest import mock
        from media.hw_capture import HwCapture
        sent = []

        class Hung:
            returncode = 1

            def __init__(self, *a, **k):
                self.calls = 0
                self.stdout = iter(["ERROR: no preroll\n"])   # no frame ever comes out

            def wait(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("gst", timeout)
                return 1

            def send_signal(self, sig):
                sent.append(sig)

            def kill(self):
                sent.append("kill")
        cap = object.__new__(HwCapture)
        cap.decoder = "v4l2h264dec"
        with mock.patch("media.hw_capture.subprocess.Popen", Hung), mock.patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "hung.*after one retry"):
                cap._probe_retrying(["filesrc"])
        # each try ended after SIGINT: never SIGKILL (ghost sessions); 12/11 one retry after a hang
        self.assertEqual(sent, [signal.SIGINT, signal.SIGINT])


class ProbeFrameCountTests(unittest.TestCase):
    def test_the_probe_stops_itself_after_30_frames_when_eos_after_is_ignored(self):
        # 14/11: 252's GStreamer ignores identity eos-after; the probe decoded the whole file and timed out
        import signal
        from unittest import mock
        from media.hw_capture import HwCapture
        sent = []

        class Endless:
            returncode = None

            def __init__(self, *a, **k):
                self.stdout = (f"/GstIdentity:identity0: last-message = chain ******* ({1228800} bytes, dts: x)\n"
                               for _ in range(100000))

            def send_signal(self, sig):
                sent.append(sig)
                self.stdout = iter([])
                self.returncode = 0

            def wait(self, timeout=None):
                while self.returncode is None:
                    pass
                return 0
        cap = object.__new__(HwCapture)
        cap.decoder = "qtic2vdec"
        with mock.patch("media.hw_capture.subprocess.Popen", Endless):
            log = cap._probe_once(["filesrc"])
        self.assertEqual(sent, [signal.SIGINT])
        self.assertGreaterEqual(log.count("1228800 bytes"), 30)
