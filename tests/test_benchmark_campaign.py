import asyncio
import unittest

from benchmark.campaign import run_case
from benchmark.config import RunConfig


class Worker:
    def __init__(self, ready, expected=4, prepare_error=False, delay=0, corrupt=False):
        self.ready = ready
        self.expected = expected
        self.prepare_error = prepare_error
        self.delay = delay
        self.corrupt = corrupt
        self.measured = self.closed = False
    async def prepare(self, config):
        if self.prepare_error:
            raise RuntimeError("preflight failed")
        self.ready.add(config.node_id)
    async def measure(self, config):
        assert len(self.ready) == self.expected, "measurement before READY barrier"
        self.measured = True
        await asyncio.sleep(self.delay)
        return {**config.as_dict(), "completed": 10, "execution_status": "PASS",
                "run_id": "wrong" if self.corrupt else config.run_id}
    async def close(self):
        self.closed = True


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, **worker_options):
        ready = set()
        configs = [RunConfig("campaign", "run", f"edge-{i}", "model", "edge", measured_frames=10) for i in range(4)]
        workers = {c.node_id: Worker(ready, **worker_options) for c in configs}
        return configs, workers

    async def test_four_nodes_barrier_and_common_interval(self):
        configs, workers = self.setup_case()
        ticks = iter([0, 2_000_000_000])
        report = await run_case(configs, workers, clock=lambda: next(ticks))
        self.assertEqual(report["coordinated_fps"], 20)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(all(w.closed for w in workers.values()))

    async def test_failed_prepare_never_starts_partial_case(self):
        configs, workers = self.setup_case()
        workers["edge-0"].prepare_error = True
        report = await run_case(configs, workers)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertFalse(any(w.measured for w in workers.values()))
        self.assertTrue(all(w.closed for w in workers.values()))

    async def test_slow_node_timeout_not_hidden(self):
        configs, workers = self.setup_case()
        workers["edge-0"].delay = .1
        report = await run_case(configs, workers, measure_timeout_s=.01)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertIn("edge-0", report["missing_nodes"])

    async def test_cross_run_reply_is_rejected(self):
        configs, workers = self.setup_case()
        workers["edge-0"].corrupt = True
        report = await run_case(configs, workers)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertIn("identity mismatch", report["errors"]["edge-0"])

    async def test_wrong_worker_topology_rejected(self):
        configs, workers = self.setup_case()
        workers.pop("edge-0")
        with self.assertRaises(ValueError):
            await run_case(configs, workers)
