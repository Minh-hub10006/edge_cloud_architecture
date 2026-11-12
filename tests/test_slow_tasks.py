"""Slow lane step 2 (8/11): UC3 counting, UC2 congestion, their Kafka messages and database rows."""
from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from contracts import Box, FrameIdentity

FPS = 3  # results per second fed to the tasks (enough for 5 s windows)


def events(seconds, boxes_at, start_s=0.0):
    """SlowEvents of one camera, FPS per second; boxes_at(t) -> boxes of the frame at t seconds."""
    out = []
    for k in range(round(seconds * FPS)):
        t = start_s + k / FPS
        ident = FrameIdentity("edge-246", "cam-1", "s", 7, round(t * 15), 1_790_000_000_000_000_000 + round(t * 1e9),
                              round(t * 1e9))
        frame = SimpleNamespace(identity=ident, capture_mono_ns=round(t * 1e9))
        out.append(SimpleNamespace(frame=frame, result=SimpleNamespace(boxes=tuple(boxes_at(t)))))
    return out


def car(track, x, y=.5, cls=2):
    x %= .9  # keep the box inside the frame
    return Box(x, y, x + .05, y + .1, .9, cls, "car", track_id=track)


async def run_task(task, evs):
    from edge_agent.fanout import BoundedLatestQueue
    from edge_agent.slow_lane import FrameRing
    queue, stop = BoundedLatestQueue(10_000), asyncio.Event()
    for e in evs:
        queue.put_latest(e)
    stop.set()
    return await task.run(queue, FrameRing(), stop)


class CountTests(unittest.IsolatedAsyncioTestCase):
    async def test_windows_of_people_and_vehicles_with_new_tracks_and_standing_share(self):
        from edge_agent.slow_tasks import CountTask
        sent = []

        async def sink(kind, payload):
            sent.append((kind, payload))

        def boxes(t):   # 2 people (no track), car 1 standing, car 2 driving, a bus appearing at 6 s
            out = [Box(.1, .1, .2, .4, .8, 0, "person"), Box(.3, .1, .4, .4, .8, 0, "person"),
                   car(1, .2), car(2, .1 + .05 * t)]
            return out + ([car(3, .7, cls=5)] if t >= 6 else [])
        stats = await run_task(CountTask(sink), events(10.7, boxes))
        rows = [p for k, p in sent if k == "count"]
        self.assertEqual(stats["windows_sent"], len(rows))
        self.assertEqual(len(rows), 3)                          # 0-5 s, 5-10 s, and the rest at the end
        first, second = rows[0], rows[1]
        self.assertEqual((first["persons"], first["cars"], first["buses"]), (2, 2, 0))
        self.assertEqual((first["new_tracks"], second["new_tracks"]), (2, 1))   # the bus is new in window 2
        self.assertEqual(second["vehicles_max"], 3)
        self.assertAlmostEqual(first["stationary_share"], .5, delta=.1)        # one of the two cars stands
        self.assertEqual((first["node_id"], first["camera_id"], first["stream_epoch"]), ("edge-246", "cam-1", 7))


class CongestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_event_starts_after_the_hold_and_ends_after_the_hold(self):
        from edge_agent.slow_tasks import CongestionTask
        sent = []

        async def sink(kind, payload):
            sent.append((kind, payload))
        jam = lambda t: [car(i, .01 * i) for i in range(32)]               # 32 standing vehicles
        free = lambda t: [car(100 + i, (.05 * t + .01 * i) % 1) for i in range(5)]
        cfg = {"congestion_hold_s": 10}
        evs = events(5, free) + events(25, jam, 5) + events(25, free, 30)
        stats = await run_task(CongestionTask(sink, cfg), evs)
        kinds = [(k, p["phase"]) for k, p in sent]
        self.assertEqual(kinds, [("event", "start"), ("event", "end")])
        start, end = sent[0][1], sent[1][1]
        self.assertEqual(start["event_id"], end["event_id"])
        self.assertEqual(start["event_type"], "congestion")
        self.assertGreaterEqual(start["detail"]["vehicles_mean"], 30)
        self.assertGreater(end["ended_utc_ns"], start["started_utc_ns"])
        self.assertEqual(stats, {"events_started": 1, "open_at_end": False})

    async def test_many_moving_vehicles_are_not_congestion(self):
        from edge_agent.slow_tasks import CongestionTask
        sent = []

        async def sink(kind, payload):
            sent.append(payload)
        busy = lambda t: [car(i, (.1 * t + .01 * i) % 1) for i in range(35)]   # 35 vehicles, all driving
        await run_task(CongestionTask(sink, {"congestion_hold_s": 10}), events(40, busy))
        self.assertEqual(sent, [])


class StoreTests(unittest.IsolatedAsyncioTestCase):
    def test_count_and_event_messages_become_rows_of_the_schema(self):
        from pathlib import Path
        from cloud_core.results_writer import COUNT_COLUMNS, slow_row
        count = {"kind": "count", "node_id": "n", "camera_id": "c", "stream_id": "s", "stream_epoch": 1,
                 "window_end_utc_ns": 1_790_000_000_000_000_000, "window_s": 5, "frames": 15, "persons": 1.5,
                 "bicycles": 0, "cars": 3, "motorcycles": 4, "buses": 0, "trucks": 0, "vehicles_max": 9,
                 "persons_max": 2, "new_tracks": 3, "stationary_share": None}
        kind, row = slow_row(json.dumps(count).encode())
        self.assertEqual((kind, len(row)), ("count", len(COUNT_COLUMNS)))
        self.assertEqual(row[0].year, 2026)
        kind, params = slow_row(json.dumps({"kind": "event", "phase": "end", "event_id": "e", "event_type": "congestion",
                                            "node_id": "n", "camera_id": "c", "stream_id": "s", "stream_epoch": 1,
                                            "started_utc_ns": 1, "ended_utc_ns": 2, "detail": {"x": 1}}).encode())
        self.assertEqual((kind, json.loads(params[-2]), params[-1]), ("event", {"end": {"x": 1}}, None))
        import base64
        import tempfile
        with tempfile.TemporaryDirectory() as d:   # a start carries a snapshot: kept as a file, its path in the row
            _, params = slow_row(json.dumps({"kind": "event", "phase": "start", "event_id": "n/c/s/1/fall/9",
                                             "event_type": "fall", "node_id": "n", "camera_id": "c", "stream_id": "s",
                                             "stream_epoch": 1, "started_utc_ns": 1, "detail": {},
                                             "snapshot_jpeg_b64": base64.b64encode(b"jpg").decode()}).encode(), d)
            self.assertTrue(params[-1].startswith(d) and params[-1].endswith(".jpg"))
            self.assertEqual(open(params[-1], "rb").read(), b"jpg")
        schema = (Path(__file__).resolve().parents[1] / "scripts/db/schema.sql").read_text(encoding="utf-8")
        counts = schema.split("CREATE TABLE IF NOT EXISTS counts")[1].split(");")[0]
        for column in COUNT_COLUMNS:
            self.assertIn(column, counts)
        with self.assertRaises(ValueError):
            slow_row(b'{"kind": "other"}')

    async def test_edge_publishes_counts_and_events_on_their_topics_keyed_by_camera(self):
        from contracts.kafka_transport import EDGE_COUNTS_TOPIC, EDGE_EVENTS_TOPIC, KafkaCloudClient
        sent = []

        class Producer:
            async def send(self, topic, *, key, value):
                sent.append((topic, key, json.loads(value)))
        client = KafkaCloudClient("fake", node_id="edge-246")
        client.producer = Producer()
        ids = {"node_id": "edge-246", "camera_id": "cam-1", "stream_id": "s", "stream_epoch": 7}
        await client.publish_slow("count", {**ids, "persons": 1})   # no wait: goes out in the background (P5)
        await asyncio.sleep(0)
        await client.publish_slow("event", {**ids, "phase": "start"})
        await asyncio.sleep(0)
        self.assertEqual([t for t, _, _ in sent], [EDGE_COUNTS_TOPIC, EDGE_EVENTS_TOPIC])
        self.assertEqual(sent[0][1], b"edge-246/cam-1/s/7")
        self.assertEqual((sent[0][2]["kind"], sent[1][2]["kind"]), ("count", "event"))


if __name__ == "__main__":
    unittest.main()
