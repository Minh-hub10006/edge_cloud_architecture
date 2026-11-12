"""UC3 (P2, 10/11): dense trigger and gathering rule on the Edge, DM-Count pre/post-processing and the Cloud handler."""
from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace

import numpy as np

from contracts import Box, FrameIdentity

FPS = 3


def person(track, x, y=.5, h=.1, conf=.9):
    return Box(x, y - h, x + h / 3, y, conf, 0, "person", track_id=track)


def events(seconds, boxes_at, hires=True, start_s=0.0):
    out = []
    for k in range(round(seconds * FPS)):
        t = start_s + k / FPS
        ident = FrameIdentity("edge-246", "cam-1", "s", 7, round(t * 15), 1_790_000_000_000_000_000 + round(t * 1e9),
                              round(t * 1e9))
        big = np.full((1920, 1080, 3), 90, np.uint8) if hires else None
        frame = SimpleNamespace(identity=ident, capture_mono_ns=round(t * 1e9), hires=big,
                                pixels=np.full((640, 640, 3), 90, np.uint8), source_size=(1080, 1920))
        out.append(SimpleNamespace(frame=frame, result=SimpleNamespace(boxes=tuple(boxes_at(t)))))
    return out


async def run(task, evs):
    from edge_agent.fanout import BoundedLatestQueue
    from edge_agent.slow_lane import FrameRing
    q, stop = BoundedLatestQueue(10_000), asyncio.Event()
    for e in evs:
        q.put_latest(e)
    stop.set()
    return await task.run(q, FrameRing(), stop)


class DensityCodeTests(unittest.TestCase):
    def test_square_input_matches_the_measured_variant_and_padding_never_counts(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import crowd_eval
        from inference.crowd_density import prepare_square, summarize
        bgr = np.random.default_rng(3).integers(0, 255, (1920, 1080, 3), np.uint8)
        x, (h, w) = prepare_square(bgr)
        self.assertEqual((x.shape, (h, w)), ((1, 3, 1280, 1280), (1280, 720)))
        self.assertTrue(np.allclose(x, crowd_eval.prepare(bgr, "sq 1280")))   # the variant scored MAE 8.95
        d = np.zeros((1, 1, 160, 160), np.float32)
        d[0, 0, :160, :90] = .01          # the image area: 720 / 8 = 90 columns
        d[0, 0, :, 90:] = 5               # padding: must not count
        s = summarize(d, (h, w))
        self.assertAlmostEqual(s["count"], 144.0, places=1)
        self.assertEqual((len(s["grid"]), len(s["grid"][0])), (8, 8))
        self.assertAlmostEqual(sum(map(sum, s["grid"])), 144.0, places=0)


class CrowdTriggerTests(unittest.IsolatedAsyncioTestCase):
    def task(self, sent, **cfg):
        from edge_agent.crowd_task import CrowdTask

        async def sink(kind, payload):
            sent.append((kind, payload))
        return CrowdTask(sink, cfg)

    async def test_dense_scenes_send_one_big_frame_every_5_s_with_a_deadline(self):
        sent = []
        crowd = lambda t: [person(i, .02 * i) for i in range(35)]          # 35 people >= 30
        stats = await run(self.task(sent), events(12, crowd))
        frames = [p for k, p in sent if k == "crowd"]
        self.assertEqual(len(frames), 3)                                    # t = 0, 5, 10 s
        p = frames[0]
        self.assertEqual((p["width"], p["height"]), (720, 1280))            # long side 1280, camera aspect
        self.assertEqual(p["deadline_utc_ns"] - p["capture_utc_ns"], 10_000_000_000)
        self.assertEqual(p["persons_boxes"], 35)
        import cv2
        img = cv2.imdecode(np.frombuffer(base64.b64decode(p["jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (1280, 720))
        self.assertEqual(stats["sent"], 3)

    async def test_overlapping_boxes_also_mean_dense_and_sparse_scenes_send_nothing(self):
        sent = []
        merged = lambda t: [person(i, .3 + .005 * i) for i in range(10)]   # 10 people, heavily overlapping
        sparse = lambda t: [person(i, .09 * i) for i in range(10)]
        await run(self.task(sent), events(3, sparse))
        self.assertEqual(sent, [])
        await run(self.task(sent), events(3, merged))
        self.assertEqual(len(sent), 1)

    async def test_the_cloud_throttle_spaces_the_frames(self):
        sent = []
        crowd = lambda t: [person(i, .02 * i) for i in range(35)]
        await run(self.task(sent, throttle=lambda: 2.0), events(12, crowd))
        self.assertEqual(len(sent), 2)                                      # t = 0, 10 s

    async def test_without_full_resolution_the_analytics_frame_is_sent_at_the_camera_aspect(self):
        sent = []
        crowd = lambda t: [person(i, .02 * i) for i in range(35)]
        stats = await run(self.task(sent), events(1, crowd, hires=False))
        self.assertEqual((sent[0][1]["width"], sent[0][1]["height"]), (360, 640))
        self.assertEqual(stats["sent_low_res"], 1)

    async def test_a_dense_frame_without_its_own_full_copy_uses_the_newest_one_in_the_ring(self):
        # 10/11 on 246: full resolution on every 5th frame; frames 3-4 after one fell back to 640 (-6..-16 people)
        from edge_agent.fanout import BoundedLatestQueue
        from edge_agent.slow_lane import FrameRing
        sent = []
        task = self.task(sent)
        crowd = lambda t: [person(i, .02 * i) for i in range(35)]
        evs = events(2, crowd)
        ring = FrameRing()
        for k, e in enumerate(evs[:3]):   # frame 0 has its full copy; frame 2 (0.67 s later, the dense one) does not
            e.frame.hires = e.frame.hires if k == 0 else None
            ring.add(e.frame)
        q, stop = BoundedLatestQueue(10), asyncio.Event()
        q.put_latest(evs[2])
        stop.set()
        stats = await task.run(q, ring, stop)
        self.assertEqual((sent[0][1]["width"], sent[0][1]["height"]), (720, 1280))   # full resolution, not 640
        self.assertEqual(sent[0][1]["frame_id"], evs[0].frame.identity.source_frame_id)
        self.assertEqual(stats["sent_low_res"], 0)


class GatheringTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_standing_group_starts_an_event_and_a_walking_crowd_does_not(self):
        from edge_agent.crowd_task import GatheringTask
        sent = []

        async def sink(kind, payload):
            sent.append(payload)
        group = lambda t: [person(i, .4 + .03 * i) for i in range(6)]                 # 6 close, standing
        walkers = lambda t: [person(i, (.1 + .15 * i + .05 * t) % .9) for i in range(6)]   # spread out, walking
        await run(GatheringTask(sink, {"gathering_hold_s": 10}), events(30, walkers))
        self.assertEqual(sent, [])
        await run(GatheringTask(sink, {"gathering_hold_s": 10}), events(30, group))
        self.assertEqual([m["phase"] for m in sent], ["start"])
        self.assertEqual(sent[0]["event_type"], "gathering")
        self.assertGreaterEqual(sent[0]["detail"]["group_max"], 5)


class CloudCrowdTests(unittest.IsolatedAsyncioTestCase):
    async def test_rows_for_every_frame_and_one_alert_per_dense_episode(self):
        import cv2
        from cloud_core import uc3_crowd
        uc3_crowd._episodes.clear()
        jpeg = base64.b64encode(cv2.imencode(".jpg", np.zeros((1280, 720, 3), np.uint8))[1].tobytes()).decode()
        msg = {"node_id": "edge-246", "camera_id": "cam-1", "stream_id": "s", "stream_epoch": 7, "frame_id": 1,
               "capture_utc_ns": 1_790_000_000_000_000_000, "persons_boxes": 33, "jpeg_b64": jpeg}
        people = [150.0]

        def model(x):
            d = np.zeros((1, 1, 160, 160), np.float32)
            d[0, 0, 10, 10] = people[0]
            return d
        out = []
        for _ in range(uc3_crowd.HOLD):
            out += await uc3_crowd.crowd_frame(msg, model)
        rows = [v for t, _, v in out if t == uc3_crowd.OUT_TOPIC]
        alerts = [v for t, _, v in out if t == uc3_crowd.EVENTS_TOPIC]
        self.assertEqual(len(rows), uc3_crowd.HOLD)
        self.assertEqual(rows[0]["persons_est"], 150.0)
        self.assertEqual([a["phase"] for a in alerts], ["start"])
        self.assertEqual((alerts[0]["event_type"], alerts[0]["snapshot_jpeg_b64"]), ("crowd", jpeg))
        self.assertEqual(alerts[0]["detail"]["dense_cells"][0][:2], [0, 0])   # the crowd is in the top-left cell
        ended = await uc3_crowd.close_silent(now=10 ** 9)                       # the Edge stopped sending
        self.assertEqual([v["phase"] for _, _, v in ended], ["end"])
        self.assertEqual(uc3_crowd._episodes, {})

    def test_crowd_rows_fit_the_table(self):
        from pathlib import Path
        from cloud_core.results_writer import CROWD_COLUMNS, slow_row
        row = {"kind": "crowd", "node_id": "n", "camera_id": "c", "stream_id": "s", "stream_epoch": 1, "frame_id": 2,
               "capture_utc_ns": 1_790_000_000_000_000_000, "persons_boxes": 3, "persons_est": 40.5,
               "grid": [[0.0] * 8] * 8, "infer_ms": 20.0}
        kind, values = slow_row(json.dumps(row).encode())
        self.assertEqual((kind, len(values)), ("crowd", len(CROWD_COLUMNS)))
        schema = (Path(__file__).resolve().parents[1] / "scripts/db/schema.sql").read_text(encoding="utf-8")
        table = schema.split("CREATE TABLE IF NOT EXISTS crowd")[1].split(");")[0]
        for column in CROWD_COLUMNS:
            self.assertIn(column, table)


if __name__ == "__main__":
    unittest.main()
