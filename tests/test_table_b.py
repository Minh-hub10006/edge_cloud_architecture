"""scripts/table_b.py: which runs and windows enter the measured table."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("table_b", Path(__file__).resolve().parents[1] / "scripts" / "table_b.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)

MANIFEST = {"fps": 25, "total_frames": 100, "sequences": [{"sequence": "A", "start_frame": 0, "frames": 100}]}


def window(i, status="accepted"):
    return {"stream_key": ["edge-x", "c", "s", 1], "window_id": i,
            "raw": {"timely_fps": 15.0, "effective_fps": 15.0, "encoder_write_ms": 10.0},
            "context": {"model_mode": "rtdetr_cloud", "target_fps": 15.0, "decision_id": "d"},
            "frames": [{"frame_id": i * 15, "status": status, "placement": "cloud"}]}


class TableBTests(unittest.TestCase):
    def run_dir(self, root, name, status, windows, start=None):
        run = Path(root) / "edge-x" / name
        run.mkdir(parents=True)
        (run / "verdict.json").write_text(json.dumps({"status": status}))
        (run / "state.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
        if start is not None:
            (run / "start_frame").write_text(str(start))

    def test_pass_runs_kept_short_review_runs_dropped_long_ones_cut_at_gpu_fault(self):
        with tempfile.TemporaryDirectory() as root:
            self.run_dir(root, "a_pass", "PASS", [window(0), window(1)], start=50)
            self.run_dir(root, "b_aborted", "REVIEW", [window(0)])
            cut = [window(i) for i in range(tb.MIN_WINDOWS)]
            cut[5] = window(5, "inference_error")
            self.run_dir(root, "c_gpu_fault", "REVIEW", cut)
            rows = list(tb.windows(root, MANIFEST))
        by_run = {}
        for r in rows:
            by_run.setdefault(r["run"], []).append(r["window"])
        self.assertEqual(by_run, {"a_pass": [0, 1], "c_gpu_fault": [0, 1, 2, 3, 4]})
        self.assertEqual(rows[0]["scene"]["second"], 2)  # start_frame 50 -> clip frame 50 = second 2
        self.assertAlmostEqual(rows[0]["encoder_load"], .15)


if __name__ == "__main__":
    unittest.main()
