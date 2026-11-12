"""scripts/sim_replay.py: measured model sequences replayed through drl.sim."""
import importlib.util
import unittest
from pathlib import Path

from test_drl_sim import table  # tests/ is on sys.path under discovery

spec = importlib.util.spec_from_file_location("sim_replay", Path(__file__).resolve().parents[1] / "scripts" / "sim_replay.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def row(run, edge, window, mode, timely, encoder_load=.04):
    return {"run": run, "edge": edge, "window": window, "mode": mode, "timely": timely, "latency_ms": 20.0,
            "encoder_load": encoder_load, "cpu_max": 20.0}


class SimReplayTests(unittest.TestCase):
    def test_replay_follows_each_runs_modes_and_scores_the_timely_error(self):
        rows = [row("20261013_1600_a", "edge-247", w, "yolo_edge" if w < 3 else "rtdetr_cloud", 1.0) for w in range(6)]
        # 252 at the afternoon encoder load (1.09): the replay infers ~21 FPS of encoder -> timely .82 in sim.
        rows += [row("20261013_1600_b", "edge-252", w, "yolo_edge", .5, encoder_load=1.09) for w in range(6)]
        out = sr.replay(rows, table())
        self.assertEqual({k: len(v["live"]) for k, v in out.items()},
                         {("edge-247", "yolo_edge"): 3, ("edge-247", "rtdetr_cloud"): 3, ("edge-252", "yolo_edge"): 6})
        _lines, errors = sr.summarize(out)
        self.assertLess(errors[("edge-247", "yolo_edge")], .05)       # 247 keeps 15 FPS on the Edge
        self.assertAlmostEqual(errors[("edge-252", "yolo_edge")], .32, delta=.03)  # sim .82 vs fake .5


if __name__ == "__main__":
    unittest.main()
