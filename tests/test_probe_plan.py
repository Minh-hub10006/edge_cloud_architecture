"""drl.probe_plan: selective probing stays within the fixed scheme's budget and skips streams on D-FINE-X."""
import unittest
from types import SimpleNamespace

from drl.probe_plan import plan


def stream(ema, age=0.0, var=1.0):
    return SimpleNamespace(probe_ema=ema, probe_age=age, probe_var=var)


class ProbePlanTests(unittest.TestCase):
    def setUp(self):
        self.profile = (1.0, -.03)  # Edge F1 = 1 - .03 x probe: .70 at probe 10
        self.cfg = SimpleNamespace(decision_s=5)

    def test_budget_is_one_second_per_stream_not_on_dfine_and_dfine_gets_none(self):
        sim = SimpleNamespace(streams=[stream(10), stream(2), stream(20), stream(10)], cfg=self.cfg)
        acts = [("yolo11s_edge", 15), ("yolo11s_edge", 15), ("rtdetr_cloud", 15), ("dfine_x_cloud", 15)]
        p = plan(sim, acts, self.profile)
        self.assertEqual(sum(p.values()), 3)
        self.assertNotIn(3, p)

    def test_borderline_and_unknown_streams_go_first(self):
        sim = SimpleNamespace(streams=[stream(10), stream(0), stream(None), stream(30)], cfg=self.cfg)
        p = plan(sim, [("yolo11s_edge", 15)] * 4, self.profile)
        self.assertGreaterEqual(p.get(0, 0), p.get(1, 0))  # F1 estimate .70 (borderline) vs 1.0 (clearly easy)
        self.assertGreaterEqual(p.get(2, 0), 1)  # never probed
        self.assertLessEqual(max(p.values()), 5)


if __name__ == "__main__":
    unittest.main()
