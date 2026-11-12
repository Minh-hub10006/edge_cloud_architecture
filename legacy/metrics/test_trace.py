from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.telemetry.trace import FrameTrace, JsonlTraceWriter, STAGES


class FrameTraceTests(unittest.TestCase):
    def test_schema_contains_every_stage(self):
        trace = FrameTrace("run-1", 7, "yolo", backend="snpe_daemon")
        data = trace.to_dict()
        self.assertEqual(set(data["durations_ms"]), set(STAGES))
        self.assertEqual(data["mode"], "YOLO")

    def test_local_span_and_external_duration(self):
        trace = FrameTrace("run-1", 8, "dfine")
        with trace.span("frame_read"):
            pass
        trace.observe_ms("cloud_decode", 2.5)
        trace.finish(payload_bytes=1024)
        data = trace.to_dict()
        self.assertGreaterEqual(data["durations_ms"]["frame_read"], 0.0)
        self.assertEqual(data["durations_ms"]["cloud_decode"], 2.5)
        self.assertEqual(data["attributes"]["payload_bytes"], 1024)

    def test_jsonl_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            trace = FrameTrace("run-1", 9, "blend")
            trace.finish()
            with JsonlTraceWriter(path) as writer:
                writer.write(trace)
                self.assertEqual(writer.count, 1)
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["frame_id"], 9)
            self.assertEqual(row["mode"], "BLEND")

    def test_jsonl_writer_accepts_resource_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resource.jsonl"
            with JsonlTraceWriter(path) as writer:
                writer.write({"wall_time_ns": 1, "cpu_utilization_ratio": 0.5})
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["cpu_utilization_ratio"], 0.5)

    def test_unknown_stage_is_rejected(self):
        trace = FrameTrace("run-1", 10, "yolo")
        with self.assertRaises(KeyError):
            trace.start("not-a-stage")


if __name__ == "__main__":
    unittest.main()
