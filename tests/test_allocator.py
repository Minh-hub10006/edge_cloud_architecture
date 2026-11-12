"""drl.allocator: joint assignment from per-stream scores never overfills the Cloud."""
import dataclasses
import unittest

import numpy as np

from drl.allocator import allocate, cloud_capacity
from drl.sim_v3 import FPS_LEVELS, MODES, TABLE_NAMES, SimV3, SimV3Config, TableV3

ACTIONS = [(m, f) for m in MODES for f in FPS_LEVELS]


def sim_with(n, **cfg):
    rows = [{"sequence": "S0", "window": w, "split": "test", "model": m, "fps": f, "f1": .8}
            for w in range(30) for m in TABLE_NAMES for f in FPS_LEVELS]
    sim = SimV3(TableV3(rows, "test"), SimV3Config(vn_share=0, **cfg), 0)
    sim.reset(n)
    return sim


def prefer(n, best, edge=("yolo11s_edge", 15)):
    """Every stream scores `best` highest, `edge` second, the rest low."""
    s = np.zeros((n, len(ACTIONS)))
    s[:, ACTIONS.index(best)] = 1.0
    s[:, ACTIONS.index(edge)] = .5
    return s


class AllocatorTests(unittest.TestCase):
    def test_never_overfills_and_the_rest_stay_on_their_best_edge_action(self):
        sim = sim_with(14)
        acts = allocate(sim, prefer(14, ("dfine_x_cloud", 15)), ACTIONS, gpu_cap=.9)
        caps, _ = cloud_capacity(sim)
        on = [a for a in acts if a[0] == "dfine_x_cloud"]
        self.assertLessEqual(sum(f for _, f in on) / caps["dfine_x_cloud"] + sim._probe_load(caps), .9 + 1e-9)
        self.assertGreater(len(on), 0)
        self.assertTrue(all(a == ("yolo11s_edge", 15) for a in acts if a[0] != "dfine_x_cloud"))

    def test_no_gain_keeps_everyone_on_the_edge(self):
        sim = sim_with(6)
        acts = allocate(sim, prefer(6, ("yolo_edge", 10)), ACTIONS)
        self.assertEqual(acts, [("yolo_edge", 10)] * 6)

    def test_heavy_ocr_admits_fewer_cloud_streams(self):
        light = sim_with(14, ocr_read_ms=50.0)
        heavy = sim_with(14, ocr_read_ms=50.0)
        light.downstream, heavy.downstream = 0.0, .8  # OCR service: 4 busy workers x .2 of the GPU
        n_light = sum(m == "dfine_x_cloud" for m, _ in allocate(light, prefer(14, ("dfine_x_cloud", 15)), ACTIONS))
        n_heavy = sum(m == "dfine_x_cloud" for m, _ in allocate(heavy, prefer(14, ("dfine_x_cloud", 15)), ACTIONS))
        self.assertLess(n_heavy, n_light)

    def test_highest_gain_streams_go_first(self):
        sim = sim_with(14)
        s = prefer(14, ("dfine_x_cloud", 15))
        s[3, ACTIONS.index(("dfine_x_cloud", 15))] = 5.0  # stream 3 gains most
        acts = allocate(sim, s, ACTIONS, gpu_cap=.2)  # room for one or two streams only
        self.assertEqual(acts[3], ("dfine_x_cloud", 15))


if __name__ == "__main__":
    unittest.main()
