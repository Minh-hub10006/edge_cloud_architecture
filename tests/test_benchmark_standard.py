"""Loop source, staged model timing, GPU sampling and the aggregate report."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from benchmark import report
from benchmark.config import RunConfig
from benchmark.model import infer_staged, measure_model
from benchmark.resources import sample_gpu


class StagedDetector:
    model_id = "staged"

    def infer_detailed(self, frame):
        time.sleep(.02)  # exceed Windows' coarse monotonic clock
        return [], 3.0, {"cloud_preprocess_ms": 1.0, "cloud_session_call_ms": 1.5, "cloud_postprocess_ms": .5}


class PreparedDetector:
    model_id = "prepared"

    def prepare_frame(self, frame):
        return frame

    def infer_prepared(self, prepared):
        return [], 2.0, {"edge_preprocess_ms": .5, "edge_session_call_ms": 1.5}


class PlainDetector:
    model_id = "plain"

    def infer(self, frame):
        return [], 1.0


class StagedTimingTests(unittest.TestCase):
    def test_uses_adapter_stage_split_when_available(self):
        self.assertIn("cloud_session_call_ms", infer_staged(StagedDetector(), 0)[2])
        self.assertIn("edge_session_call_ms", infer_staged(PreparedDetector(), 0)[2])
        self.assertEqual(infer_staged(PlainDetector(), 0)[2], {})

    def test_model_capacity_reports_component_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig("c", "r", "edge", "staged", "cloud", 5, 1)
            summary = measure_model(config, StagedDetector(), [0, 1], Path(tmp) / "run",
                                    identity={"artifact_sha256": "h", "video_sha256": "h",
                                              "model_id": "staged", "runtime": "cuda"}, sample_s=0)
        self.assertEqual(summary["execution_status"], "PASS")
        self.assertEqual(summary["stages_ms"]["cloud_session_call_ms"]["count"], 5)


class LoopSourceTests(unittest.TestCase):
    def test_loop_replays_clip_with_increasing_identity(self):
        cv2 = __import__("cv2")
        import numpy as np
        from media.opencv_source import OpenCVFrameSource
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            for i in range(5):
                writer.write(np.full((48, 64, 3), i * 40, np.uint8))
            writer.release()

            async def read(loop):
                source = OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s",
                                           max_frames=12, realtime_file=False, loop=loop)
                frames = [frame async for frame in source]
                return source, frames
            source, frames = asyncio.run(read(True))
            _, plain = asyncio.run(read(False))
        self.assertEqual(len(plain), 5)
        self.assertEqual(len(frames), 12)
        self.assertEqual(source.loops_completed, 2)
        ids = [f.identity.source_frame_id for f in frames]
        pts = [f.identity.source_pts for f in frames]
        self.assertEqual(ids, list(range(12)))
        self.assertTrue(all(b > a for a, b in zip(pts, pts[1:])))


class LateConsumerSourceTests(unittest.TestCase):
    def test_a_late_consumer_gets_the_newest_frame_and_numbers_keep_the_files(self):
        # 12/11 on 252: analytics at 9-14 fps read the 15 fps file frame by frame and fell 3-21 s behind the video
        cv2 = __import__("cv2")
        import numpy as np
        from media.opencv_source import OpenCVFrameSource
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
            for i in range(60):
                writer.write(np.full((48, 64, 3), i * 4, np.uint8))
            writer.release()

            async def read():
                source = OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s", max_frames=60)
                ids = []
                async for frame in source:
                    ids.append(frame.identity.source_frame_id)
                    await asyncio.sleep(0.05)   # 20 fps consumer, 50 fps file
                return source, ids
            source, ids = asyncio.run(read())
        self.assertGreater(source.frames_skipped, 20)
        self.assertLess(len(ids), 40)
        self.assertEqual(ids, sorted(set(ids)))               # increasing, never repeated
        self.assertLess(ids[-1], 60)
        self.assertEqual(len(ids) + source.frames_skipped, ids[-1] + 1)   # every number is the file's own


class StartFrameSourceTests(unittest.TestCase):
    def test_start_frame_skips_into_the_clip_and_loops_back_to_its_start(self):
        cv2 = __import__("cv2")
        import numpy as np
        from media.opencv_source import OpenCVFrameSource
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            for i in range(5):
                writer.write(np.full((48, 64, 3), i * 40, np.uint8))
            writer.release()

            async def read(start, frames=5):
                source = OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s", max_frames=frames,
                                           realtime_file=False, loop=True, start_frame=start)
                return [(f.identity.source_frame_id, round(float(f.pixels.mean()) / 40)) async for f in source]
            frames = asyncio.run(read(3))
            with self.assertRaises(RuntimeError):
                asyncio.run(read(9))
        self.assertEqual(frames, [(0, 3), (1, 4), (2, 0), (3, 1), (4, 2)])
        with self.assertRaises(ValueError):
            OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s", start_frame=-1)


class PrefetchSourceTests(unittest.TestCase):
    def test_prefetch_yields_same_frames_and_stops_early(self):
        cv2 = __import__("cv2")
        import numpy as np
        from media.opencv_source import OpenCVFrameSource
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            for i in range(5):
                writer.write(np.full((48, 64, 3), i * 40, np.uint8))
            writer.release()

            async def read(prefetch, limit=None):
                source = OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s", max_frames=12,
                                           realtime_file=False, loop=True, prefetch=prefetch)
                out = []
                async for frame in source:
                    out.append((frame.identity.source_frame_id, frame.identity.source_pts,
                                frame.pixels.tobytes()))
                    if limit and len(out) == limit:
                        break  # consumer stops early: reader thread must exit
                return out
            plain, ahead = asyncio.run(read(0)), asyncio.run(read(3))
            early = asyncio.run(read(3, limit=2))
        self.assertEqual([f[:2] for f in plain], [f[:2] for f in ahead])
        self.assertEqual([f[2] for f in plain], [f[2] for f in ahead])
        self.assertEqual(len(early), 2)
        import threading
        self.assertFalse(any(t.name == "frame-prefetch" for t in threading.enumerate()))
        with self.assertRaises(ValueError):
            OpenCVFrameSource(path, node_id="n", camera_id="c", stream_id="s", prefetch=2)


class GpuSampleTests(unittest.TestCase):
    def test_missing_tool_is_unknown_not_zero(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertIsNone(sample_gpu())

    def test_parses_query(self):
        done = mock.Mock(stdout="87, 5120, 24564, 61, 310.5\n")
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch("subprocess.run", return_value=done):
            self.assertEqual(sample_gpu(), {"utilization_pct": 87.0, "memory_used_mib": 5120.0,
                                            "memory_total_mib": 24564.0, "temperature_c": 61.0,
                                            "power_w": 310.5})


def _run(root, name, fps, status="PASS", inflight=4, stages=None, completed=1000, reasons=None):
    directory = root / name
    directory.mkdir(parents=True)
    summary = {"model_key": "m", "profile": "pipeline_capacity", "transport": "grpc",
               "measured_frames": 1000, "completed": completed, "failure_reasons": reasons or {},
               "repetition": 1, "execution_status": status,
               "effective_fps": fps, "bytetrack_completion_fps": fps,
               "stages_ms": stages or {"edge_session_call_ms": {"p50": 20, "p95": 25},
                                       "edge_postprocess_ms": {"p50": 5, "p95": 6},
                                       "e2e_ms": {"p50": 40, "p95": 60}},
               "measurement_interval": {"started_mono_ns": 0, "finished_mono_ns": 10}}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "manifest.json").write_text(json.dumps({"options": {"max_inflight": inflight},
                                                         "identity": {"source_loop": True}}))


class ReportTests(unittest.TestCase):
    def test_gates_and_bottleneck_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, fps in enumerate((50, 51, 49)):
                _run(root, f"stable{i}", fps)
            for i, fps in enumerate((30, 45, 60)):
                _run(root, f"unstable{i}", fps, inflight=8)
            _run(root, "few", 40, inflight=2)
            _run(root, "bad", 40, status="FAIL", inflight=1)
            table = {row["max_inflight"]: row for row in report.aggregate(report.load_runs([root]))}
            report.write(list(table.values()), root / "out")
            self.assertTrue((root / "out" / "report.md").is_file())
        self.assertEqual(table[4]["gate"], "PASS")
        self.assertEqual(table[8]["gate"], "UNSTABLE")
        self.assertEqual(table[2]["gate"], "PROVISIONAL")
        self.assertEqual(table[1]["gate"], "FAIL")
        self.assertEqual([s["stage"] for s in table[4]["top_stages"]],
                         ["edge_session_call_ms", "edge_postprocess_ms"])
        self.assertAlmostEqual(table[4]["e2e_p95_ms"], 60)


class ThermalReportTests(unittest.TestCase):
    def test_reports_hottest_sensor_inside_measurement_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                _run(root, f"r{i}", 50)
                rows = [{"mono_ns": 5, "host_cpu_percent": 40, "thermal_raw": {"a": 70000, "b": 81000 + i * 1000}},
                        {"mono_ns": 99, "thermal_raw": {"a": 99000}}]  # outside the 0..10 ns window
                (root / f"r{i}" / "resources-edge.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (row,) = report.aggregate(report.load_runs([root]))
            report.write([row], root / "out")
            text = (root / "out" / "report.md").read_text(encoding="utf-8")
        self.assertAlmostEqual(row["edge_temp_max_c"], 83.0)
        self.assertAlmostEqual(row["edge_cpu_mean_pct"], 40.0)
        self.assertIn("| 40.0 | 83.0 | PASS |", text)


class BottleneckTests(unittest.TestCase):
    def test_lowest_cap_lane_is_bottleneck(self):
        stages = [{"stage": "edge_session_call_ms", "p50_ms": 20.0}, {"stage": "edge_postprocess_ms", "p50_ms": 5.0},
                  {"stage": "edge_preprocess_ms", "p50_ms": 8.0}, {"stage": "cloud_bytetrack_update_ms", "p50_ms": 2.0}]
        rows = report.lanes(stages, measured_fps=38.0)
        self.assertEqual(rows[0]["lane"], "edge_infer")
        self.assertAlmostEqual(rows[0]["fps_cap"], 40.0)
        self.assertAlmostEqual(rows[0]["load"], .95)
        self.assertEqual([r["lane"] for r in rows], ["edge_infer", "edge_producer", "cloud_tracker"])


class ShellLineEndingTests(unittest.TestCase):
    def test_shell_scripts_are_lf(self):
        # Linux bash fails on CRLF ("$'\r': command not found"); Git Bash hides it on Windows.
        root = Path(__file__).resolve().parents[1] / "scripts"
        self.assertEqual([p.name for p in root.rglob("*.sh") if b"\r" in p.read_bytes()], [])


def _load_ledger():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "results_ledger.py"
    spec = importlib.util.spec_from_file_location("results_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LedgerTests(unittest.TestCase):
    def test_ledger_keeps_hosts_apart(self):
        results_ledger = _load_ledger()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for host, fps in (("edge-246", 50), ("edge-252", 40)):
                scope = root / host / "v31/runs/bench/20261011_120000/yolov8n/pipeline/10000x3"
                for i in range(3):
                    _run(scope, f"r{i}", fps)
            for i in range(3):
                _run(root / "edge-247/v33/runs/bench/contention_3edge_20261011_193000/rtdetrv2/pipeline",
                     f"contention-rep{i + 1}-inflight2", 70)
            lossy = root / "edge-252/v36/runs/bench/contention_3edge_20261011_193000/lwdetr/pipeline"
            _run(lossy, "contention-rep1-inflight2", 14, status="FAIL", completed=964,
                 reasons={"cloud:media_disconnected: frame=1 cache=...": 30, "cloud:media_disconnected: frame=2": 6})
            _run(lossy, "contention-rep2-inflight2", 17)
            lines, rows = results_ledger.build(root)
        self.assertTrue(any(line.startswith("| contention_3edge_20261011_193000 | edge-247 |") and "| 3/3 | 70.0 |" in line
                            and "| 100.0% | - |" in line for line in lines))
        # 1964 of 2000 frames; reasons grouped by category without per-frame detail.
        self.assertTrue(any(line.startswith("| contention_3edge_20261011_193000 | edge-252 |")
                            and "| 98.2% | cloud:media_disconnected x36 | FAIL |" in line for line in lines))
        rows = [r for r in rows if not r["campaign"].startswith("contention_")]
        self.assertEqual([(r["host"], r["effective_fps_mean"]) for r in rows], [("edge-246", 50), ("edge-252", 40)])
        self.assertTrue(any(line.startswith("### edge-252 · v31 · 20261011_120000 ·") for line in lines))


    def test_ratio_uses_newest_campaign_not_release_name(self):
        results_ledger = _load_ledger()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for release, campaign, fps in (("opt4_v21", "20261010_233200", 54.7),
                                           ("edge_video_platform_v33", "20261011_185432", 101.6)):
                for i in range(3):
                    _run(root / "edge-246" / release / "runs/bench" / campaign / "yolov8n/pipeline/10000x3", f"r{i}", fps)
                for path in (root / "edge-246" / release).rglob("summary.json"):
                    path.write_text(path.read_text().replace('"measured_frames": 1000', '"measured_frames": 10000'))
            lines, _ = results_ledger.build(root)
        (row,) = [line for line in lines if line.startswith("| edge-246 | yolov8n |")]
        self.assertIn("| 101.6 (4) |", row)


if __name__ == "__main__":
    unittest.main()
