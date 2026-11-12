from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmark.config import RunConfig, validate_campaign
from benchmark.model import measure_model, preload_video, sha256
from benchmark.summarize import coordinated_summary, stats, summarize
from benchmark.trace import JsonlWriter, frame_event
from benchmark.__main__ import main


def config(**changes):
    return RunConfig(**{**dict(campaign_id="campaign", run_id="run", node_id="edge-246",
                              model_key="yolo-test", placement="edge", measured_frames=4,
                              warmup_frames=2), **changes})


class FakeDetector:
    model_id = "test-model"
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at
    def infer(self, frame):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected model failure")
        return [object(), object()], 2.0


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"artifact_sha256": "a" * 64, "video_sha256": "b" * 64,
                         "model_id": "test-model", "runtime": {"backend": "test"}}

    def run_model(self, cfg=None, detector=None, **kwargs):
        ticks = iter(range(0, 100_000_000, 1_000_000))
        return measure_model(cfg or config(), detector or FakeDetector(), ["frame0", "frame1"],
                             self.root / "result", identity=self.identity, sample_s=0,
                             clock=lambda: next(ticks), **kwargs)

    def test_config_rejects_bad_bounds_and_profiles(self):
        for changes in ({"measured_frames": 0}, {"measured_frames": True},
                        {"warmup_frames": -1}, {"transport": "grpc"}, {"placement": "invalid"},
                        {"node_id": " "}, {"profile": "pipeline_capacity", "transport": "none"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    def test_campaign_accepts_multiple_distinct_edges(self):
        data = {"schema_version": 1, "nodes": [
            {"node_id": f"edge-{i}", "video": "/video.mp4"} for i in range(4)],
            "cloud": {"node_id": "cloud"},
            "cases": [{"model_key": "yolo", "placement": "edge", "profile": "realtime", "transport": "kafka"}]}
        self.assertIs(validate_campaign(data), data)
        data["nodes"].append(data["nodes"][0])
        with self.assertRaises(ValueError):
            validate_campaign(data)

    def test_warmup_excluded_and_adapter_latency_not_fps(self):
        detector = FakeDetector()
        report = self.run_model(detector=detector)
        self.assertEqual(detector.calls, 6)
        self.assertEqual(report["completed"], 4)
        self.assertEqual(report["detections_total"], 8)
        self.assertEqual(report["warmup_observations"], 2)
        self.assertEqual(report["stages_ms"]["adapter_reported_ms"]["count"], 4)
        self.assertEqual(report["elapsed_s"], .009)
        self.assertAlmostEqual(report["effective_fps"], 4 / .009)
        self.assertEqual(report["execution_status"], "PASS")
        self.assertEqual(report["quality_status"], "NOT_EVALUATED")
        self.assertTrue((self.root / "result/stages.csv").is_file())

    def test_measured_failure_stays_visible(self):
        report = self.run_model(detector=FakeDetector(fail_at=4))
        self.assertEqual(report["frames"], {"ok": 3, "error": 1})
        self.assertEqual(report["execution_status"], "FAIL")
        self.assertTrue(report["gates"]["measurement"])
        self.assertEqual(report["stages_ms"]["adapter_call_ms"]["count"], 3)

    def test_warmup_failure_aborts_without_false_pass(self):
        report = self.run_model(detector=FakeDetector(fail_at=1))
        self.assertEqual(report["completed"], 0)
        self.assertEqual(report["warmup_errors"], 1)
        self.assertIn("warmup failed", report["fatal_error"])
        self.assertEqual(report["execution_status"], "FAIL")

    def test_previous_result_is_never_overwritten(self):
        self.run_model()
        with self.assertRaises(FileExistsError):
            self.run_model()

    def test_missing_identity_is_not_accepted(self):
        self.identity["runtime"] = None
        self.assertEqual(self.run_model()["execution_status"], "FAIL")

    def test_stats_empty_percentiles_nonfinite(self):
        self.assertEqual(stats([])["mean"], None)
        self.assertEqual(stats([1, 2, 3, 4])["p50"], 2.5)
        self.assertAlmostEqual(stats([1, 2, 3, 4])["p95"], 3.85)
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                stats([value])

    def test_writer_round_trip_and_closed_writer(self):
        path = self.root / "trace.jsonl"
        writer = JsonlWriter(path)
        writer.write({"status": "test"})
        writer.close()
        self.assertEqual(writer.written, 1)
        self.assertEqual(json.loads(path.read_text()), {"status": "test"})
        with self.assertRaises(RuntimeError):
            writer.write({})
        with self.assertRaises(FileExistsError):
            JsonlWriter(path)

    def test_writer_queue_full_fails_not_drops(self):
        writer = JsonlWriter(self.root / "trace.jsonl")
        with patch.object(writer._queue, "put_nowait", side_effect=queue.Full):
            with self.assertRaisesRegex(RuntimeError, "measurement invalid"):
                writer.write({})
        writer.close()

    def test_writer_error_makes_run_invalid(self):
        with patch.object(JsonlWriter, "write", side_effect=RuntimeError("logger error")):
            report = self.run_model()
        self.assertEqual(report["execution_status"], "FAIL")
        self.assertIn("logger error", report["fatal_error"])

    def test_event_rejects_nan_and_invalid_timing(self):
        with self.assertRaises(ValueError):
            frame_event(config(), frame_id=1, phase="measure", status="ok", started_ns=2,
                        finished_ns=1, durations_ms={})
        with self.assertRaises(ValueError):
            frame_event(config(), frame_id=1, phase="measure", status="ok", started_ns=1,
                        finished_ns=2, durations_ms={"x": float("nan")})

    def test_mixed_trace_identity_and_duplicate_rejected(self):
        cfg = config(warmup_frames=0, measured_frames=2)
        row = frame_event(cfg, frame_id=0, phase="measure", status="ok", started_ns=1,
                          finished_ns=2, durations_ms={"x": None})
        path = self.root / "trace.jsonl"
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(path, cfg, 1)
        row["node_id"] = "other"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "mixed trace"):
            summarize(path, cfg, 1)

    def test_missing_stage_is_not_zero(self):
        cfg = config(warmup_frames=0, measured_frames=1)
        row = frame_event(cfg, frame_id=0, phase="measure", status="ok", started_ns=1,
                          finished_ns=2, durations_ms={"unavailable": None})
        path = self.root / "trace.jsonl"
        path.write_text(json.dumps(row) + "\n")
        report = summarize(path, cfg, 1)
        self.assertIsNone(report["stages_ms"]["unavailable"]["mean"])
        self.assertEqual(report["stages_ms"]["unavailable"]["missing"], 1)

    def test_coordinated_fps_uses_common_interval_and_missing_node(self):
        one = {**config().as_dict(), "completed": 20, "execution_status": "PASS", "effective_fps": 100}
        two = {**one, "node_id": "edge-247", "completed": 10, "effective_fps": 300}
        result = coordinated_summary([one, two], expected_nodes=["edge-246", "edge-247"], elapsed_s=10)
        self.assertEqual(result["coordinated_fps"], 3)
        self.assertEqual(result["status"], "PASS")
        result = coordinated_summary([one], expected_nodes=["edge-246", "edge-247"], elapsed_s=10)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing_nodes"], ["edge-247"])
        with self.assertRaises(ValueError):
            coordinated_summary([one, one], expected_nodes=["edge-246"], elapsed_s=10)
        with self.assertRaises(ValueError):
            coordinated_summary([one, {**two, "run_id": "other"}],
                                expected_nodes=["edge-246", "edge-247"], elapsed_s=10)

    def test_preload_memory_is_bounded_and_capture_released(self):
        class Frame:
            nbytes = 12
        class Capture:
            released = False
            def isOpened(self): return True
            def read(self): return True, Frame()
            def release(self): self.released = True
        cap = Capture()
        with patch("cv2.VideoCapture", return_value=cap):
            with self.assertRaisesRegex(ValueError, "cache exceeds"):
                preload_video(Path("fake.mp4"), 3, 15)
        self.assertTrue(cap.released)

    def test_hash_uses_file_bytes(self):
        path = self.root / "file"
        path.write_bytes(b"abc")
        self.assertEqual(sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_no_warmup_is_supported(self):
        report = self.run_model(cfg=config(warmup_frames=0))
        self.assertEqual(report["warmup_observations"], 0)
        self.assertEqual(report["completed"], 4)

    def test_supervisor_timeout_keeps_incomplete_record(self):
        args = ["model", "--adapter=yolo_snpe", "--artifact=fake.dlc", "--video=fake.mp4",
                "--model-key=test", "--node-id=edge", "--out", str(self.root)]
        with patch("benchmark.__main__.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            self.assertEqual(main(args), 1)
        files = list(self.root.glob("supervised-*/supervisor.json"))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text())
        self.assertEqual(record["state"], "TIMEOUT_INCOMPLETE")

    def test_measured_interval_is_available_for_resource_filtering(self):
        result = self.run_model()
        interval = result["measurement_interval"]
        self.assertEqual(interval["clock_scope"], "edge-246")
        self.assertAlmostEqual((interval["finished_mono_ns"] - interval["started_mono_ns"]) / 1e9,
                               result["elapsed_s"])

    def test_failed_stage_durations_are_not_hidden(self):
        result = self.run_model(detector=FakeDetector(fail_at=3))
        self.assertEqual(result["failed_stages_ms"]["adapter_call_ms"]["count"], 1)
        self.assertTrue(result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
