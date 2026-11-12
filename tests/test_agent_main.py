"""The Edge Agent's own start-up path (edge_agent.control_v1.run), with the network, video and models faked.

10/11: a variable of the camera profile shadowed the media profile and every agent failed at start, with or without a
slow lane; no test ran run(args). These do, up to the pipeline call, and check what the pipeline receives.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class FakeClient:
    def __init__(self, *a, **k):
        self.slow_throttle = 1.0

    async def start(self):
        pass

    async def get_feedback(self):
        return {"feedback": 1}

    async def publish_slow(self, kind, payload, wait=False):
        pass

    async def close(self):
        pass


def run_agent(extra, profile=None):
    from edge_agent import control_v1
    seen = {}

    async def pipeline(source, publisher, client, detectors, controller, monitor, **kw):
        seen.update(kw, source=source, publisher=publisher)
        return {"ok": True}

    with tempfile.TemporaryDirectory() as d:
        args = ["--video", "v.mp4", "--main-video", "main.mp4", "--mode", "rtdetr_cloud", "--transport", "kafka",
                "--node-id", "edge-246", "--camera-id", "cam-1", "--frames", "15", "--metrics-port", "0",
                "--outbox-dir", d, *extra]
        if profile is not None:
            Path(d, "cams.json").write_text(json.dumps(profile))
            args += ["--camera-profile", str(Path(d, "cams.json"))]
        parsed = control_v1.parser().parse_args(args)
        monitor = SimpleNamespace(feedback=SimpleNamespace(accept=lambda v: None))
        with mock.patch("contracts.control_client_v1.KafkaControlClient", FakeClient), \
             mock.patch.object(control_v1, "run_pipeline", pipeline), \
             mock.patch.object(control_v1, "TimeWindowMonitor", lambda *a, **k: monitor), \
             mock.patch("media.opencv_source.OpenCVFrameSource", lambda *a, **k: SimpleNamespace(kw=k)), \
             mock.patch("media.copy_publisher.CopyPublisher", lambda *a, **k: SimpleNamespace(kw=k)), \
             mock.patch("edge_agent.resources_v2.ResourceSampler", lambda: None):
            asyncio.run(control_v1.run(parsed))
    return seen


class AgentStartTests(unittest.TestCase):
    def test_starts_without_a_slow_lane(self):
        seen = run_agent([])
        self.assertIsNone(seen["slow_lane"])
        self.assertEqual(seen["source"].kw["hires_every"], 0)          # no full-resolution branch without UC1/UC3
        self.assertEqual(seen["publisher"].kw["main_video"], "main.mp4")

    def test_a_camera_profile_chooses_tasks_thresholds_and_zones(self):
        road = [[0, .5], [1, .5], [1, 1], [0, 1]]
        seen = run_agent([], profile={"edge-246/cam-1": {"tasks": ["count", "congestion"],
                                                        "config": {"congestion_vehicles": 12},
                                                        "zones": {"road": road}}})
        lane = seen["slow_lane"]
        self.assertEqual([t.name for t, _ in lane.tasks], ["count", "congestion"])
        congestion = dict((t.name, t) for t, _ in lane.tasks)["congestion"]
        self.assertEqual((congestion.cfg["congestion_vehicles"], congestion.cfg["zones"]["road"]), (12, road))
        self.assertEqual(congestion.cfg["throttle"](), 1.0)
        self.assertIsNotNone(lane.sink.outbox)                          # events go through the durable outbox
        self.assertEqual(seen["source"].kw["hires_every"], 0)          # count/congestion need no full resolution

    def test_slow_tasks_flag_wins_over_the_profile(self):
        seen = run_agent(["--slow-tasks", "check"], profile={"default": {"tasks": ["count"]}})
        self.assertEqual([t.name for t, _ in seen["slow_lane"].tasks], ["check"])
        self.assertEqual(seen["source"].kw["hires_every"], 5)          # check looks at full-resolution frames


if __name__ == "__main__":
    unittest.main()
