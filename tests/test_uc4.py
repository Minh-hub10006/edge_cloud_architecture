"""UC4 (P3, 11/11): Edge rules (fight clip, crash crop, loitering) and the Cloud checks with stand-in models."""
from __future__ import annotations

import asyncio
import base64
import unittest
from types import SimpleNamespace

import numpy as np

from contracts import Box, FrameIdentity

FPS = 10


def box(cls, track, x, y, w, h, conf=.9):
    return Box(x, y - h, x + w, y, conf, cls, "x", track_id=track)


def events(seconds, boxes_at, hires=True):
    out = []
    for k in range(round(seconds * FPS)):
        t = k / FPS
        ident = FrameIdentity("edge-246", "cam-1", "s", 7, k, 1_790_000_000_000_000_000 + round(t * 1e9), round(t * 1e9))
        frame = SimpleNamespace(identity=ident, capture_mono_ns=round(t * 1e9),
                                hires=np.full((1920, 1080, 3), 90, np.uint8) if hires else None,
                                pixels=np.full((640, 640, 3), 90, np.uint8), source_size=(1080, 1920))
        out.append(SimpleNamespace(frame=frame, result=SimpleNamespace(boxes=tuple(boxes_at(t)))))
    return out


async def run(task, evs):
    """Feed the events one by one, the ring filled as the slow lane does (frames before their results)."""
    from edge_agent.fanout import BoundedLatestQueue
    from edge_agent.slow_lane import FrameRing
    ring, stop = FrameRing(), asyncio.Event()
    q = BoundedLatestQueue(10_000)
    for e in evs:
        ring.add(e.frame)
        q.put_latest(e)
    stop.set()
    return await task.run(q, ring, stop)


def sink_into(sent):
    async def sink(kind, payload):
        sent.append((kind, payload))
    return sink


class FightRuleTests(unittest.IsolatedAsyncioTestCase):
    def jitter(self, t, k):
        return .02 * np.sin(t * 9 + k)   # fast back-and-forth, as people hitting each other

    async def test_two_agitated_people_on_foot_send_one_clip_of_crops(self):
        from edge_agent.disorder_task import FightTask
        sent = []
        people = lambda t: [box(0, 1, .40 + self.jitter(t, 0), .6, .03, .1), box(0, 2, .44 + self.jitter(t, 1), .6, .03, .1)]
        stats = await run(FightTask(sink_into(sent)), events(5, people))
        self.assertEqual([k for k, _ in sent], ["action"])          # one clip: the next waits fight_every_s (10 s)
        p = sent[0][1]
        self.assertEqual((p["check"], p["persons"], sorted(p["tracks"])), ("fight", 2, [1, 2]))
        self.assertGreaterEqual(len(p["frames"]), 5)
        import cv2
        sizes = {cv2.imdecode(np.frombuffer(base64.b64decode(f["jpeg_b64"]), np.uint8), 1).shape for f in p["frames"]}
        self.assertEqual(len(sizes), 1)                              # every crop the same size (pose -> action input)
        self.assertEqual(stats["clips"], 1)

    async def test_riders_and_people_standing_still_send_nothing(self):
        from edge_agent.disorder_task import FightTask
        sent = []
        riders = lambda t: [box(3, 10, .4 + .1 * t, .65, .05, .06), box(0, 1, .41 + .1 * t + self.jitter(t, 0), .6, .03, .1),
                            box(3, 11, .44 + .1 * t, .65, .05, .06), box(0, 2, .45 + .1 * t + self.jitter(t, 1), .6, .03, .1)]
        still = lambda t: [box(0, 1, .40, .6, .03, .1), box(0, 2, .44, .6, .03, .1)]
        walking = lambda t: [box(0, 1, .10 + .1 * t, .6, .03, .1), box(0, 2, .14 + .1 * t, .6, .03, .1)]
        await run(FightTask(sink_into(sent)), events(5, riders))
        await run(FightTask(sink_into(sent)), events(5, still))
        await run(FightTask(sink_into(sent)), events(5, walking))   # side by side, fast, straight
        self.assertEqual(sent, [])


class CrashRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_moving_bike_that_ends_lying_still_sends_one_crop(self):
        from edge_agent.disorder_task import CrashTask
        sent = []

        def bike(t):   # rides 2 s (upright: tall in pixels), then lies on the road (wide), still
            if t < 2:
                return [box(3, 5, .3 + .05 * t, .5 + .1 * t, .03, .05)]
            return [box(3, 5, .4, .72, .09, .02)]
        stats = await run(CrashTask(sink_into(sent)), events(8, bike))
        self.assertEqual([k for k, _ in sent], ["crash"])
        self.assertEqual((sent[0][1]["track_id"], sent[0][1]["full_res"]), (5, True))
        self.assertEqual(stats["sent"], 1)

    async def test_a_parked_bike_lying_since_the_start_is_not_a_crash(self):
        from edge_agent.disorder_task import CrashTask
        sent = []
        await run(CrashTask(sink_into(sent)), events(8, lambda t: [box(3, 5, .4, .72, .09, .02)]))
        self.assertEqual(sent, [])


class LoiterRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_someone_staying_starts_an_event_that_ends_when_they_leave(self):
        from edge_agent.disorder_task import LoiterTask
        sent = []
        stay = lambda t: [box(0, 7, .5, .6, .03, .1)] if t < 20 else []
        passer = lambda t: [box(0, 8, .05 * t, .6, .03, .1)]   # walks 0.5 box heights/s
        await run(LoiterTask(sink_into(sent), {"loiter_s": 10, "loiter_gone_s": 2}), events(25, stay))
        self.assertEqual([p["phase"] for _, p in sent], ["start", "end"])
        self.assertEqual(sent[0][1]["event_type"], "loitering")
        sent.clear()
        await run(LoiterTask(sink_into(sent), {"loiter_s": 10}), events(18, passer))
        self.assertEqual(sent, [])


def jpeg(h=200, w=120):
    import cv2
    return base64.b64encode(cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))[1].tobytes()).decode()


def fake_pose(standing=True, people_count=2):
    """RTMO-like outputs: `people_count` people in 640-input pixels, upright or lying."""
    def pose(x):
        b = x.shape[0]
        dets = np.zeros((b, people_count, 5), np.float32)
        kps = np.zeros((b, people_count, 17, 3), np.float32)
        for i in range(people_count):
            x0 = 100 + 150 * i
            dets[:, i] = [x0, 100, x0 + 60, 400, .9]
            kps[:, i, :, 2] = .9
            if standing:
                kps[:, i, :, 0], kps[:, i, :, 1] = x0 + 30, np.linspace(110, 390, 17)
            else:
                kps[:, i, :, 0], kps[:, i, :, 1] = np.linspace(x0, x0 + 280, 17), 300
        return dets, kps
    return pose


class CloudUc4Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from cloud_core import uc4_disorder
        uc4_disorder._episodes.clear()
        self.m = {"node_id": "edge-246", "camera_id": "cam-1", "stream_id": "s", "stream_epoch": 7, "frame_id": 3,
                  "capture_utc_ns": 1_790_000_000_000_000_000}

    async def test_a_clip_scored_as_fighting_starts_one_event(self):
        from cloud_core import uc4_disorder
        seen = []

        def action(x):
            seen.append(x.shape)
            prob = np.full((1, 60), .1 / 57, np.float32)
            prob[0, [49, 50, 51]] = .3
            return prob
        msg = {**self.m, "frames": [{"jpeg_b64": jpeg()} for _ in range(10)], "persons": 2, "region": [0, 0, 1, 1]}
        out = await uc4_disorder.action_clip(msg, fake_pose(), action)
        out += await uc4_disorder.action_clip(msg, fake_pose(), action)       # the same fight: no second start
        rows = [v for t, _, v in out if t == uc4_disorder.OUT_TOPIC]
        events = [v for t, _, v in out if t == uc4_disorder.EVENTS_TOPIC]
        self.assertEqual(seen, [(1, 2, 100, 17, 3)] * 2)
        self.assertAlmostEqual(rows[0]["score"], .9, places=3)
        self.assertEqual([(e["event_type"], e["phase"]) for e in events], [("fight", "start")])
        self.assertIsNotNone(events[0]["snapshot_jpeg_b64"])
        ended = await uc4_disorder.close_quiet(now=10 ** 9)
        self.assertEqual([v["phase"] for _, _, v in ended], ["end"])

    async def test_a_calm_clip_gives_a_row_and_no_event(self):
        from cloud_core import uc4_disorder
        calm = lambda x: np.eye(60, dtype=np.float32)[[0]]                     # "drink water"
        msg = {**self.m, "frames": [{"jpeg_b64": jpeg()} for _ in range(6)]}
        out = await uc4_disorder.action_clip(msg, fake_pose(), calm)
        self.assertEqual([t for t, _, _ in out], [uc4_disorder.OUT_TOPIC])

    async def test_a_crash_needs_someone_lying(self):
        from cloud_core import uc4_disorder
        msg = {**self.m, "jpeg_b64": jpeg(300, 300), "track_id": 5}
        up = await uc4_disorder.crash_frame(msg, fake_pose(standing=True, people_count=1))
        down = await uc4_disorder.crash_frame(msg, fake_pose(standing=False, people_count=1))
        self.assertEqual([t for t, _, _ in up], [uc4_disorder.OUT_TOPIC])
        self.assertEqual([v["event_type"] for t, _, v in down if t == uc4_disorder.EVENTS_TOPIC], ["crash"])

    def test_follow_keeps_each_person_in_their_slot(self):
        from cloud_core.uc4_disorder import follow
        def p(x):
            kp = np.zeros((17, 3), np.float32)
            kp[:, 0], kp[:, 2] = x, 1
            return (.9, np.array([x - 5, 0, x + 5, 10], np.float32), kp)
        frames = [[p(40), p(60)], [p(62), p(41)], [], [p(43)]]           # listed in a different order, one missing
        seq = follow(frames, np.array([50, 5]))
        self.assertEqual([s[:, 0, 0].tolist() for s in seq], [[40, 60], [41, 62], [], [43]])

    def test_the_slow_worker_registers_uc4(self):
        import subprocess
        import sys
        from pathlib import Path
        out = subprocess.run([sys.executable, "-B", "-m", "cloud_core.slow_worker", "--list"], capture_output=True,
                             text=True, cwd=Path(__file__).resolve().parents[1], timeout=120)
        self.assertIn("edge.action-clip.v1", out.stdout, out.stderr)
        self.assertIn("edge.crash-frame.v1", out.stdout)
        self.assertIn("close_quiet", out.stdout)


class SkeletonInputTests(unittest.TestCase):
    def test_short_clips_loop_to_100_frames_and_unused_person_slots_stay_zero(self):
        from inference.action_skeleton import prepare_clip
        kp = np.zeros((1, 17, 3), np.float32)
        kp[0, :, :2], kp[0, :, 2] = (60, 100), .8
        x = prepare_clip([kp, kp * 0, kp], (200, 120))
        self.assertEqual(x.shape, (1, 2, 100, 17, 3))
        self.assertTrue(np.allclose(x[0, 0, 0, 0], [0, 0, .8]))         # image centre -> (0, 0)
        self.assertTrue(np.allclose(x[0, 0, 1, 0], [-1, -1, 0]))        # an empty frame in a used slot: as MMAction2
        self.assertTrue(np.allclose(x[0, 0, 3], x[0, 0, 0]))             # frame 3 = frame 0 (looped)
        self.assertFalse(x[0, 1].any())                                  # second person never seen



class StoreTests(unittest.TestCase):
    def test_action_rows_fit_the_actions_table(self):
        # 12/11: cloud.action.v1 (check scores) was not stored
        import json
        from pathlib import Path
        from cloud_core.results_writer import ACTION_COLUMNS, slow_row
        row = {"schema_version": 1, "kind": "action", "node_id": "n", "camera_id": "c", "stream_id": "s",
               "stream_epoch": 1, "capture_utc_ns": 1_790_000_000_000_000_000, "frame_id": 3, "check": "fight",
               "score": 0.12, "frames": 10, "frames_with_people": 9, "infer_ms": 40.0}
        kind, values = slow_row(json.dumps(row).encode())
        self.assertEqual((kind, len(values)), ("action", len(ACTION_COLUMNS)))
        got = dict(zip(ACTION_COLUMNS, values))
        self.assertEqual((got["check_type"], got["score"]), ("fight", 0.12))
        self.assertEqual(json.loads(got["detail"]), {"frames": 10, "frames_with_people": 9})
        schema = (Path(__file__).resolve().parents[1] / "scripts/db/schema.sql").read_text(encoding="utf-8")
        table = schema.split("CREATE TABLE IF NOT EXISTS actions")[1].split(");")[0]
        for column in ACTION_COLUMNS:
            self.assertIn(column, table)

    def test_prune_removes_only_old_evidence_files(self):
        import os
        import tempfile
        from cloud_core.media_store import prune
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "events"))
            old, new, other = (os.path.join(d, "events", n) for n in ("old.jpg", "new.jpg", "notes.txt"))
            for f in (old, new, other):
                open(f, "w").close()
            os.utime(old, (0, 0))
            os.utime(other, (0, 0))
            self.assertEqual(prune(d, keep_days=30), 1)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "events"))), ["new.jpg", "notes.txt"])


if __name__ == "__main__":
    unittest.main()
