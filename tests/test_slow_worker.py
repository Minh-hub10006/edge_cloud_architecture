"""P1 (10/11): Cloud slow-lane worker rules (late, handlers) and the lag -> throttle controller."""
from __future__ import annotations

import json
import unittest


class LagTests(unittest.TestCase):
    def test_throttle_doubles_when_behind_and_halves_back_when_caught_up(self):
        from cloud_core.slow_feedback import LagController
        lag = LagController("pose@test", high_s=8, low_s=2, period_s=2)
        now_ns = 1_000_000_000_000
        throttles = []
        for t, age_s in enumerate([10, 10, 10, 10, 5, 1, 1, 1, 1]):   # one period per age
            for _ in range(20):
                lag.observe(now_ns - round(age_s * 1e9), now_ns)
            m = lag.tick(now=lag.next_at)
            throttles.append(m["throttle"])
        self.assertEqual(throttles, [2, 4, 8, 8, 8, 4, 2, 1, 1])   # capped at 8; holds while between low and high
        self.assertEqual(m["source"], "pose@test")
        self.assertIsNone(lag.tick(now=lag.next_at - 1))           # nothing before the period ends
        self.assertEqual(lag.tick(now=lag.next_at)["messages"], 0)  # an idle period is still reported


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_messages_are_dropped_before_any_model_and_handlers_route_by_topic(self):
        from cloud_core.slow_worker import process
        calls = []

        async def crowd(m):
            calls.append(m["id"])
            return [("cloud.crowd.v1", None, {"id": m["id"]})]
        handlers = {"edge.crowd-frame.v1": crowd}
        now = 10_000_000_000
        fresh = json.dumps({"id": 1, "capture_utc_ns": now - 1, "deadline_utc_ns": now + 5}).encode()
        late = json.dumps({"id": 2, "capture_utc_ns": now - 20, "deadline_utc_ns": now - 1}).encode()
        self.assertEqual(await process("edge.crowd-frame.v1", fresh, handlers, now_ns=now),
                         ("ok", [("cloud.crowd.v1", None, {"id": 1})]))
        self.assertEqual(await process("edge.crowd-frame.v1", late, handlers, now_ns=now), ("late", []))
        self.assertEqual(await process("edge.other.v1", fresh, handlers, now_ns=now), ("no_handler", []))
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()


class EntryPointTests(unittest.TestCase):
    def test_run_as_a_module_the_worker_sees_every_use_case(self):
        # 10/11 on 249: `python -m cloud_core.slow_worker` exited with "no slow-lane handlers registered"
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run([sys.executable, "-B", "-m", "cloud_core.slow_worker", "--list"], cwd=root,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        listed = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertIn("edge.crowd-frame.v1", listed["topics"])
        self.assertIn("close_silent", listed["sweeps"])
