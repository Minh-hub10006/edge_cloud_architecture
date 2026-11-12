"""Plate path: read policy, Edge->Cloud wire, OCR service over real gRPC, Edge branch end to end (fake models)."""
import asyncio
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from cloud_core.plate_ocr_service import _init, start_server
from contracts.plate_ocr_wire import PlateOcrClient, pack, unpack
from edge_agent.plate_branch import PlateBranch, crop_plate
from edge_agent.plate_policy import PlateReadPolicy

BOX = (100, 100, 160, 140, .9)


def moved(box, dx):
    return (box[0] + dx, box[1], box[2] + dx, box[3], box[4])


class PolicyTests(unittest.TestCase):
    def reads(self, policy, frames, box=BOX, reply=None):
        out = []
        for f in range(frames):
            for t in policy.update(f, [moved(box, f)]):
                out.append(f)
                if reply is not None:
                    policy.record(t, reply)
        return out

    def test_new_plate_read_at_once_then_every_k_frames_until_cap(self):
        p = PlateReadPolicy(every=3, cap=4, confirm=0)
        self.assertEqual(self.reads(p, 30), [0, 3, 6, 9])
        self.assertEqual(p.close()[0]["end"], "cap")

    def test_no_cap_no_confirm_reads_whole_track(self):  # the team repo's main.py
        p = PlateReadPolicy(every=3, cap=0, confirm=0)
        self.assertEqual(len(self.reads(p, 45)), 15)

    def test_two_equal_valid_reads_stop(self):
        p = PlateReadPolicy(every=3, cap=10, confirm=2)
        self.assertEqual(self.reads(p, 30, reply="29E74852"), [0, 3])
        s = p.close()[0]
        self.assertEqual((s["end"], s["text"], s["reads"]), ("confirmed", "29E74852", 2))

    def test_invalid_reads_do_not_confirm(self):
        p = PlateReadPolicy(every=3, cap=5, confirm=2)
        self.assertEqual(len(self.reads(p, 30, reply="ZG")), 5)
        self.assertEqual(p.close()[0]["end"], "cap")

    def test_cap_reached_then_confirmed_reports_confirmed(self):
        p = PlateReadPolicy(every=1, cap=2, confirm=2)
        ts = p.update(0, [BOX]) + p.update(1, [BOX])
        for t in ts:
            p.record(t, "29E74852")
        self.assertEqual(p.close()[0]["end"], "confirmed")

    def test_lost_after_max_age_and_new_track_after(self):
        p = PlateReadPolicy(every=3, cap=0, confirm=0, max_age=5)
        p.update(0, [BOX])
        p.update(10, [BOX])  # 10 frames unseen > max_age: old track finished, new one read at once
        self.assertEqual([s["track"] for s in p.finished], [0])
        self.assertEqual(p.tracks[0].track_id, 1)
        self.assertEqual(p.tracks[0].reads, 1)

    def test_small_plates_tracked_not_read_and_one_box_per_track(self):
        p = PlateReadPolicy(min_px=20)
        self.assertEqual(p.update(0, [(0, 0, 30, 10, .9)]), [])
        self.assertEqual(len(p.tracks), 1)
        two = p.update(1, [BOX, (102, 101, 161, 141, .8)])  # overlapping boxes -> two tracks, not one
        self.assertEqual(len({t.track_id for t in two}), 2)

    def test_cancel_returns_the_read(self):
        p = PlateReadPolicy(every=1, cap=1, confirm=0)
        t = p.update(0, [BOX])[0]
        p.cancel(t)
        self.assertEqual((t.reads, t.end), (0, None))
        self.assertEqual(len(p.update(1, [BOX])), 1)


class WireTests(unittest.TestCase):
    def test_roundtrip_and_bad_framing(self):
        h, j = unpack(pack({"track": 3}, b"\xff\xd8jpeg"))
        self.assertEqual((h, j), ({"track": 3}, b"\xff\xd8jpeg"))
        for bad in (b"", b"\x00\x00\x10\x00{}", b"\x00\x00\x00\x02[]"):
            with self.assertRaises(ValueError):
                unpack(bad)


class FakeReader:
    def read(self, crop):
        return ("29E74852" if crop.shape[1] >= 40 else "ZG"), 3


class FakeDetector:
    """One plate moving right; frames carry their index in pixel [0, 0, 0]."""
    def __call__(self, pixels):
        f = int(pixels[0, 0, 0])
        return [moved(BOX, f)] if f < 40 else []


def frame(i):
    px = np.zeros((300, 400, 3), np.uint8)
    px[0, 0, 0] = i
    return SimpleNamespace(identity=SimpleNamespace(source_frame_id=i), pixels=px)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = ThreadPoolExecutor(2, initializer=_init, initargs=(FakeReader,))
        self.lines = []
        lines = self.lines

        class Log:
            def write(self, s): lines.append(json.loads(s))
            def flush(self): pass
        self.server, port = await start_server("127.0.0.1:0", self.pool, log=Log())
        self.client = PlateOcrClient(f"127.0.0.1:{port}", timeout_s=3)

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.stop(0)
        self.pool.shutdown()

    async def test_service_reads_and_logs_each_crop(self):
        import cv2
        jpeg = cv2.imencode(".jpg", np.full((40, 60, 3), 200, np.uint8))[1].tobytes()
        reply = await self.client.read({"track": 7, "frame": 1}, jpeg)
        self.assertEqual((reply["text"], reply["calls"], reply["lines"]), ("29E74852", 3, 2))
        self.assertGreaterEqual(reply["wait_ms"], 0)
        self.assertEqual((self.lines[0]["track"], self.lines[0]["text"]), (7, "29E74852"))
        bad = await self.client.read({}, b"not a jpeg")
        self.assertEqual(bad["error"], "decode")

    async def test_replay_cameras_follow_recorded_schedules(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from plate_replay import camera_events, run
        sched = [(0.0, 60, 40), (0.1, 120, 30), (0.2, 60, 40)]  # 3 reads in a 1.2 s loop
        events = camera_events(sched, 0.0, 2.5)
        self.assertEqual([round(t, 1) for t, _, _ in events], [0.0, 0.1, 0.2, 1.2, 1.3, 1.4, 2.4])
        self.assertEqual(camera_events(sched, 0.15, 1.0)[0][0], 0.2 - 0.15)
        crops = [np.full((40, 60, 3), 200, np.uint8)]
        s = await asyncio.wait_for(run(self.client, [sched], crops, cameras=3, seconds=0.5), 10)
        # cameras start at random points of their loop: 1-3 reads each in 0.5 s, every one answered
        self.assertTrue(3 <= s["sent"] <= 9, s)
        self.assertEqual((s["replies"], s["errors"], s["dropped"]), (s["sent"], 0, 0))
        self.assertEqual(len([x for x in self.lines if x["stream"][0].startswith("emu-")]), s["sent"])

    async def test_edge_branch_end_to_end(self):
        lines = []

        class Log:
            def write(self, s): lines.append(json.loads(s))
            def flush(self): pass
        branch = PlateBranch(FakeDetector(), PlateReadPolicy(every=3, cap=5, confirm=2), self.client,
                             stream_key=("edge", "cam", "s", 1), log=Log(), queue_capacity=64)
        done = asyncio.Event()
        task = asyncio.create_task(branch.run(done))
        for i in range(60):
            branch.queue.put_latest(frame(i))
            await asyncio.sleep(.005)
        done.set()
        summary = await asyncio.wait_for(task, 5)
        reads = [x for x in lines if x["event"] == "read"]
        track = [x for x in lines if x["event"] == "track"]
        self.assertEqual(summary["frames"], 60)
        self.assertEqual(len(track), 1)
        self.assertEqual(track[0]["end"], "confirmed")
        self.assertEqual(track[0]["text"], "29E74852")
        # replies arrive while frames keep coming, so 2-3 crops leave before the 2nd equal read lands
        self.assertTrue(2 <= len(reads) <= 3, len(reads))
        self.assertEqual(summary["crops_sent"], len(reads))
        self.assertTrue(all(r["stream"] == ["edge", "cam", "s", 1] and r["rtt_ms"] > 0 for r in reads))


class OfflinePrefixTests(unittest.TestCase):
    def test_stop_index_matches_the_live_policy(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from plate_load_offline import stop_index
        texts = ["ZG", "29E74852", "29E7485", "29E74852", "29E74852"]
        for cap, confirm in ((0, 0), (1, 0), (3, 2), (5, 2), (5, 1), (10, 3)):
            p = PlateReadPolicy(every=1, cap=cap, confirm=confirm)
            live = 0
            for f, text in enumerate(texts):
                for t in p.update(f, [BOX]):
                    p.record(t, text)
                    live += 1
            self.assertEqual(stop_index(texts, cap, confirm), (live, p.close()[0]["end"]), (cap, confirm))


class OutsideBoxTests(unittest.IsolatedAsyncioTestCase):
    async def test_box_outside_the_frame_is_skipped_not_fatal(self):
        class Ocr:
            async def read(self, header, jpeg): return {"text": "29E74852", "calls": 3}
        lines = []

        class Log:
            def write(self, s): lines.append(json.loads(s))
            def flush(self): pass
        detect = lambda px: [(900, 900, 960, 940, .9), BOX]  # noqa: E731 - first box is beyond a 400x300 frame
        branch = PlateBranch(detect, PlateReadPolicy(every=3, cap=5, confirm=2), Ocr(),
                             stream_key=("e", "c", "s", 1), log=Log(), queue_capacity=8)
        done = asyncio.Event()
        task = asyncio.create_task(branch.run(done))
        for i in range(3):
            branch.queue.put_latest(frame(i))
            await asyncio.sleep(.01)
        done.set()
        summary = await asyncio.wait_for(task, 5)
        self.assertEqual(summary["frames"], 3)
        self.assertGreaterEqual(summary["crops_empty"], 1)
        self.assertEqual(summary["crops_sent"], 1)
        self.assertTrue(any(x["event"] == "empty_crop" for x in lines))


class CropTests(unittest.TestCase):
    def test_crop_padded_and_clipped(self):
        px = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(crop_plate(px, (10, 20, 30, 40)).shape[:2], (30, 30))
        self.assertEqual(crop_plate(px, (-5, 90, 20, 120)).shape[:2], (15, 25))


@unittest.skipUnless(os.environ.get("PLATE_OCR_HOME"), "set PLATE_OCR_HOME to the installed team OCR model")
class RealReaderTests(unittest.TestCase):
    def test_calls_per_crop_follow_lines_and_passes(self):
        from inference.plate_ocr import PlateReader
        r = PlateReader(os.environ["PLATE_OCR_HOME"], passes=3, gpu=False)
        two = np.full((45, 60, 3), 230, np.uint8)
        one = np.full((30, 120, 3), 230, np.uint8)
        self.assertEqual(r.read(two)[1], 6)
        self.assertEqual(r.read(one)[1], 3)


if __name__ == "__main__":
    unittest.main()
