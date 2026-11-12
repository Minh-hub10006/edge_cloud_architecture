import asyncio
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.config import RunConfig
from benchmark.remote import coordinate, WorkerConnection


class RemoteTests(unittest.IsolatedAsyncioTestCase):
    async def campaign(self, nodes, *, corrupt=False, missing=False):
        base = RunConfig("campaign", "initial", "coordinator", "model", "edge", 4, 2,
                         "pipeline_capacity", "grpc")
        ready = {}
        server_ready = asyncio.Event()
        port = None
        original = asyncio.start_server
        async def start(*args, **kwargs):
            nonlocal port
            server = await original(*args, **kwargs)
            port = server.sockets[0].getsockname()[1]
            server_ready.set()
            return server
        async def worker(node):
            connection = await WorkerConnection.open(f"127.0.0.1:{port}", node, 2)
            try:
                for repetition in (1, 2):
                    local = replace(base, node_id=node, repetition=repetition)
                    cfg = await connection.config(local)
                    ready.setdefault(repetition, set()).add(node)
                    await connection.ready(cfg)
                    self.assertEqual(ready[repetition], set(nodes))
                    report = {**cfg.as_dict(), "completed": 4, "execution_status": "PASS"}
                    if corrupt and node == nodes[0]:
                        report["run_id"] = "wrong"
                    await connection.done(report)
                    if corrupt:
                        break
            finally:
                await connection.close()
        with tempfile.TemporaryDirectory() as tmp, patch("benchmark.remote.asyncio.start_server", side_effect=start):
            coordinator = asyncio.create_task(coordinate(base, nodes, Path(tmp) / "campaign", port=0,
                repetitions=2, ready_timeout_s=.15 if missing else 2, measure_timeout_s=2))
            await server_ready.wait()
            tasks = [asyncio.create_task(worker(node)) for node in (nodes[:-1] if missing else nodes)]
            report = await coordinator
            await asyncio.gather(*tasks, return_exceptions=True)
            return report

    async def test_three_real_socket_workers_barrier_two_repetitions(self):
        report = await self.campaign(["edge-1", "edge-2", "edge-3"])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["reports"]), 2)
        self.assertEqual(report["reports"][0]["completed"], 12)

    async def test_missing_node_not_counted_as_smaller_campaign(self):
        report = await self.campaign(["edge-1", "edge-2"], missing=True)
        self.assertEqual(report["status"], "INCOMPLETE")

    async def test_cross_run_result_incomplete(self):
        report = await self.campaign(["edge-1", "edge-2"], corrupt=True)
        self.assertEqual(report["status"], "INCOMPLETE")

    def test_remote_config_cannot_escape_output_root(self):
        for name in ("campaign_id", "run_id", "node_id"):
            with self.assertRaises(ValueError):
                replace(RunConfig("campaign", "run", "edge", "model", "edge"), **{name: "../other"})
