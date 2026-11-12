"""UC1 slow lane (8/11): which vehicle crops get a plate read, the crop message, the Cloud OCR message and its row."""
from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace

import numpy as np

from contracts import Box, FrameIdentity


def event(i, boxes, hires=True, fps=15):
    t = i / fps
    ident = FrameIdentity("edge-246", "cam-1", "s", 7, i, 1_790_000_000_000_000_000 + round(t * 1e9), round(t * 1e9))
    big = np.full((1920, 1080, 3), 120, np.uint8) if hires else None
    frame = SimpleNamespace(identity=ident, capture_mono_ns=round(t * 1e9), hires=big)
    return SimpleNamespace(frame=frame, result=SimpleNamespace(boxes=tuple(boxes)))


def vehicle(track, size, cls=2, x=.1):
    return Box(x, .2, x + size, .2 + size, .9, cls, "car", track_id=track)


async def run(task, events):
    from edge_agent.fanout import BoundedLatestQueue
    from edge_agent.slow_lane import FrameRing
    q, stop = BoundedLatestQueue(10_000), asyncio.Event()
    for e in events:
        q.put_latest(e)
    stop.set()
    return await task.run(q, FrameRing(), stop)


class PlateTaskTests(unittest.IsolatedAsyncioTestCase):
    def task(self, sent, plates=((10, 20, 60, 40, .8),), **cfg):
        from edge_agent.plate_task import PlateTask
        calls = []

        def detector(crop):
            calls.append(crop.shape)
            return list(plates)

        async def sink(kind, payload):
            sent.append((kind, payload))
        return PlateTask(sink, cfg, detector=detector), calls

    async def test_at_most_two_reads_per_vehicle_the_second_only_when_closer(self):
        sent = []
        task, calls = self.task(sent)
        # every 5th frame has a full-resolution copy; the car grows from 10 % to 40 % of the frame over 4 s
        evs = [event(i, [vehicle(1, .1 + .3 * i / 60)], hires=i % 5 == 0) for i in range(61)]
        stats = await run(task, evs)
        reads = [p for k, p in sent if k == "plate"]
        self.assertEqual([p["read_no"] for p in reads], [1, 2])
        self.assertTrue(all(p["frame_id"] % 5 == 0 for p in reads))         # only frames with their own pixels
        self.assertGreaterEqual(reads[1]["frame_id"] - reads[0]["frame_id"], 9)   # >= 0.6 s apart
        self.assertEqual(stats["sent"], 2)
        self.assertEqual(len(calls), 2)
        p = reads[0]
        x1, y1 = round(.1 * 1080), round(.2 * 1920)   # the vehicle crop's corner in the full frame
        self.assertEqual(p["plate_box"][:2], [x1 + 10, y1 + 20])
        img = np.frombuffer(base64.b64decode(p["jpeg_b64"]), np.uint8)
        import cv2
        self.assertEqual(cv2.imdecode(img, cv2.IMREAD_COLOR).shape[:2], (30, 60))   # plate + 5 px padding

    async def test_small_or_untracked_vehicles_and_people_are_skipped(self):
        sent = []
        task, calls = self.task(sent)
        evs = [event(0, [vehicle(1, .02), Box(.1, .1, .5, .5, .9, 2, "car"), Box(.1, .1, .5, .5, .9, 0, "person",
                                                                                     track_id=9)])]
        await run(task, evs)
        self.assertEqual((sent, calls), ([], []))

    async def test_detector_budget_per_second_and_no_plate_tries(self):
        sent = []
        task, calls = self.task(sent, plates=(), plate_detects_per_s=2)
        evs = [event(i * 5, [vehicle(t, .2, x=.05 * t) for t in range(10)]) for i in range(9)]   # 3 s, 10 cars
        stats = await run(task, evs)
        self.assertLessEqual(len(calls), 2 + 2 * 3)          # a 1 s burst then 2 per s
        self.assertEqual(stats["no_plate"], len(calls))
        self.assertEqual(sent, [])

    def test_the_task_needs_a_detector(self):
        from edge_agent.plate_task import PlateTask
        with self.assertRaises(ValueError):
            PlateTask(None, {})


class PlateCloudTests(unittest.TestCase):
    def test_ocr_message_and_its_database_row(self):
        from cloud_core.plate_worker import read_message
        from cloud_core.results_writer import PLATE_COLUMNS, slow_row
        crop = {"node_id": "edge-246", "camera_id": "cam-1", "stream_id": "s", "stream_epoch": 7, "track_id": 4,
                "frame_id": 30, "capture_utc_ns": 1_790_000_000_000_000_000, "vehicle_class": 3, "read_no": 1,
                "det_conf": .8, "jpeg_b64": base64.b64encode(b"jpeg").decode()}
        out = read_message(json.dumps(crop).encode(), lambda jpeg, arrived: {"text": "29E112345", "ocr_ms": 9.5,
                                                                            "wait_ms": 1.0})
        self.assertEqual((out["text"], out["valid"], out["track_id"]), ("29E112345", True, 4))
        self.assertFalse(read_message(json.dumps(crop).encode(), lambda j, a: {"text": "ABC"})["valid"])
        kind, row = slow_row(json.dumps(out).encode())
        self.assertEqual((kind, len(row)), ("plate", len(PLATE_COLUMNS)))
        from pathlib import Path
        schema = (Path(__file__).resolve().parents[1] / "scripts/db/schema.sql").read_text(encoding="utf-8")
        table = schema.split("CREATE TABLE IF NOT EXISTS plates")[1].split(");")[0]
        for column in PLATE_COLUMNS:
            self.assertIn(column, table)

    def test_edge_sends_plate_crops_on_their_topic(self):
        from contracts.kafka_transport import EDGE_PLATE_CROPS_TOPIC, KafkaCloudClient
        sent = []

        class Producer:
            async def send(self, topic, *, key, value):
                sent.append(topic)
        client = KafkaCloudClient("fake", node_id="edge-246")
        client.producer = Producer()
        asyncio.run(client.publish_slow("plate", {"node_id": "n", "camera_id": "c", "stream_id": "s", "stream_epoch": 1}))
        self.assertEqual(sent, [EDGE_PLATE_CROPS_TOPIC])


if __name__ == "__main__":
    unittest.main()


class PlateFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_plate_boxes_outside_the_vehicle_and_detector_errors_are_skipped(self):
        from edge_agent.plate_task import PlateTask
        boom = iter([RuntimeError("npu"), [(5000, 5000, 5100, 5050, .9)]])

        def detector(crop):
            item = next(boom)
            if isinstance(item, Exception):
                raise item
            return item
        task = PlateTask(None, {}, detector=detector)
        stats = await run(task, [event(0, [vehicle(1, .2)]), event(5, [vehicle(2, .2, x=.5)])])
        self.assertEqual((stats["errors"], stats["empty_crop"], stats["sent"]), (1, 1, 0))

    async def test_a_crashing_slow_task_stops_alone(self):
        from edge_agent.slow_lane import SlowLane

        class Bad:
            name = "bad"

            async def run(self, queue, ring, stop):
                raise ValueError("boom")

        class Good:
            name = "good"

            async def run(self, queue, ring, stop):
                await stop.wait()
                return {"ok": 1}
        lane = SlowLane([Bad(), Good()])
        stop = asyncio.Event()
        stop.set()
        out = await lane.run(stop)
        self.assertIn("boom", out["tasks"]["bad"]["error"])
        self.assertEqual(out["tasks"]["good"]["ok"], 1)
