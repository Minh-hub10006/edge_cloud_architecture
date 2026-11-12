"""P1 (10/11): camera profiles and zones, event episodes, the durable event outbox, and the slow-lane throttle."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from contracts import Box, FrameIdentity


class ProfileTests(unittest.TestCase):
    def test_camera_entry_merges_over_default_and_zones_are_checked(self):
        from edge_agent.camera_profile import load_profile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cams.json"
            p.write_text(json.dumps({
                "default": {"tasks": ["count"], "config": {"window_s": 5, "congestion_vehicles": 30}},
                "edge-246/cam-1": {"tasks": ["plates", "congestion"], "config": {"congestion_vehicles": 20},
                                   "zones": {"road": [[0, .5], [1, .5], [1, 1], [0, 1]]}}}))
            own = load_profile(p, "edge-246", "cam-1")
            other = load_profile(p, "edge-252", "cam-9")
            self.assertEqual(own["tasks"], ["plates", "congestion"])
            self.assertEqual(own["config"], {"window_s": 5, "congestion_vehicles": 20})
            self.assertEqual((other["tasks"], other["zones"]), (["count"], {}))
            p.write_text(json.dumps({"edge-246/cam-1": {"zones": {"road": [[0, 0], [2, 0], [1, 1]]}}}))
            with self.assertRaises(ValueError):
                load_profile(p, "edge-246", "cam-1")

    def test_an_object_is_in_a_zone_where_its_box_touches_the_ground(self):
        from edge_agent.camera_profile import in_zone
        road = [[0, .5], [1, .5], [1, 1], [0, 1]]   # lower half of the frame
        car_on_road = Box(.2, .3, .4, .7, .9, 2, "car")      # top above the line, bottom on the road
        car_above = Box(.2, .1, .4, .4, .9, 2, "car")
        self.assertTrue(in_zone(car_on_road, road))
        self.assertFalse(in_zone(car_above, road))
        self.assertTrue(in_zone(car_above, None))


def frame(t_ns):
    return SimpleNamespace(identity=FrameIdentity("edge-246", "cam-1", "s", 7, 1, 1_790_000_000_000_000_000 + t_ns, t_ns),
                           pixels=None)


class EpisodeTests(unittest.TestCase):
    def test_start_after_hold_on_end_after_hold_off_and_flicker_is_ignored(self):
        from edge_agent.events import Episode
        ep = Episode("fall", hold_on=2, hold_off=2)
        seq = [True, False, True, True, False, True, False, False, True]
        out = [ep.update(c, frame(i), {"i": i}) for i, c in enumerate(seq)]
        phases = [(i, m["phase"]) for i, m in enumerate(out) if m]
        self.assertEqual(phases, [(3, "start"), (7, "end")])
        start, end = out[3], out[7]
        self.assertEqual(start["event_id"], end["event_id"])
        self.assertIn("/fall/", start["event_id"])
        self.assertIn("snapshot_jpeg_b64", start)
        self.assertNotIn("snapshot_jpeg_b64", end)
        self.assertEqual(ep.started, 1)


class OutboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_survive_an_outage_and_a_restart_then_go_out_once_back_online(self):
        from edge_agent.events import EventOutbox, SlowSink
        online, sent = [False], []

        async def send(kind, payload, wait):
            if not online[0]:
                raise ConnectionError("Kafka down")
            sent.append((kind, payload["event_id"], payload["phase"], wait))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "edge-246_cam-1.jsonl"
            sink = SlowSink(send, EventOutbox(path))
            await sink("event", {"event_id": "e1", "phase": "start"})
            await sink("count", {"x": 1})                               # periodic data is not kept
            self.assertEqual(sink.stats["events_failed"], 1)
            self.assertEqual(list(sink.outbox.pending), ["e1#start"])
            restarted = SlowSink(send, EventOutbox(path))              # the agent restarted while offline
            self.assertEqual(list(restarted.outbox.pending), ["e1#start"])
            online[0] = True
            self.assertEqual(await restarted.resend(), 1)
            self.assertEqual(sent, [("event", "e1", "start", True)])  # waited for Kafka's ack
            self.assertEqual(EventOutbox(path).pending, {})            # acked for good
            await restarted("event", {"event_id": "e1", "phase": "end"})
            self.assertEqual(sent[-1][2], "end")
            self.assertEqual(EventOutbox(path).pending, {})


class ThrottleTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_edge_reads_the_cloud_throttle_without_breaking_its_feedback(self):
        from contracts.control_client_v1 import KafkaControlClient
        from contracts.control_v1 import FEEDBACK_TOPIC
        from contracts.kafka_transport import SLOW_FEEDBACK_TOPIC
        c = KafkaControlClient("fake", node_id="edge-246")
        msgs = [SimpleNamespace(topic=SLOW_FEEDBACK_TOPIC, value=b'{"source": "ocr", "throttle": 3}'),
                SimpleNamespace(topic=SLOW_FEEDBACK_TOPIC, value=b'{"source": "pose", "throttle": 99}'),
                SimpleNamespace(topic=SLOW_FEEDBACK_TOPIC, value=b'not json'),
                SimpleNamespace(topic=SLOW_FEEDBACK_TOPIC, value=b'{"source": "pose", "throttle": 1}')]

        class Consumer:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if not msgs:
                    raise StopAsyncIteration
                return msgs.pop(0)
        c.consumer = Consumer()
        await c._listen()
        self.assertEqual(c.slow_throttle, 3.0)   # the slowest fresh source (pose recovered to 1); bad message skipped
        c.throttles["ocr"] = (3.0, c.throttles["ocr"][1] - 31)
        self.assertEqual(c.slow_throttle, 1.0)   # a source silent for 30 s no longer counts
        self.assertIsNone(c.error)
        self.assertNotEqual(FEEDBACK_TOPIC, SLOW_FEEDBACK_TOPIC)


if __name__ == "__main__":
    unittest.main()
