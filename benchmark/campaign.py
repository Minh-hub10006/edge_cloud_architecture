"""Coordinator core for concurrent nodes; concrete remote workers come later.

Workers must make prepare/measure cancellation-safe and supervise native work in
an owned process. This module never SSHs, kills production or deploys anything.
"""
from __future__ import annotations

import asyncio
from time import monotonic_ns
from typing import Any, Protocol

from .config import RunConfig, finite_nonnegative
from .summarize import coordinated_summary


class CampaignWorker(Protocol):
    async def prepare(self, config: RunConfig) -> None:
        """Preflight + warm-up; return only when READY."""
    async def measure(self, config: RunConfig) -> dict[str, Any]:
        """START -> bounded MEASURE -> DRAIN -> final node summary."""
    async def close(self) -> None:
        """Release only resources owned by this worker/run."""


async def run_case(configs: list[RunConfig], workers: dict[str, CampaignWorker], *,
                   ready_timeout_s: float = 60, measure_timeout_s: float = 600,
                   close_timeout_s: float = 5, clock=monotonic_ns) -> dict[str, Any]:
    """Measure one simultaneous case, not a sum of independently paced FPS."""
    nodes = [config.node_id for config in configs]
    if not nodes or len(set(nodes)) != len(nodes) or set(nodes) != set(workers):
        raise ValueError("unique configs and exactly matching workers required")
    case_identity = {(c.campaign_id, c.run_id, c.model_key, c.placement, c.profile, c.transport, c.repetition)
                     for c in configs}
    if len(case_identity) != 1:
        raise ValueError("all workers must belong to the same case/repetition")
    for name, value in (("ready", ready_timeout_s), ("measure", measure_timeout_s),
                        ("close", close_timeout_s)):
        if finite_nonnegative(value, name) == 0:
            raise ValueError("timeouts must be positive")
    summaries = []
    errors = {}
    tasks = []
    started = None
    elapsed_s = 0.0
    state = "PREFLIGHT_WARMUP"
    try:
        tasks = [asyncio.create_task(workers[c.node_id].prepare(c)) for c in configs]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=ready_timeout_s)
        # No worker may start measurement until every prepare has succeeded.
        state = "MEASURE_DRAIN"
        started = clock()

        async def collect(config):
            try:
                row = await workers[config.node_id].measure(config)
                for name in ("campaign_id", "run_id", "node_id", "model_key", "placement", "profile", "transport", "repetition"):
                    if row.get(name) != getattr(config, name):
                        raise ValueError(f"worker result identity mismatch: {name}")
                if type(row.get("completed")) is not int or not 0 <= row["completed"] <= config.measured_frames:
                    raise ValueError("invalid worker completed count")
                if row.get("execution_status") not in {"PASS", "FAIL"}:
                    raise ValueError("invalid worker execution status")
                if row["execution_status"] == "PASS" and config.profile != "realtime" and row["completed"] != config.measured_frames:
                    raise ValueError("worker claimed PASS with incomplete frame count")
                if row["execution_status"] == "PASS" and config.profile == "realtime":
                    counts = row.get("frames", {})
                    if (sum(counts.values()) != config.measured_frames or counts.get("ok") != row["completed"]
                            or counts.get("error", 0) or counts.get("timeout", 0) or row["completed"] == 0):
                        raise ValueError("invalid realtime terminal frame counts")
                summaries.append(row)
            except Exception as exc:
                errors[config.node_id] = f"{type(exc).__name__}: {exc}"
                raise

        tasks = [asyncio.create_task(collect(c)) for c in configs]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=measure_timeout_s)
        state = "DONE"
    except Exception as exc:
        errors["coordinator"] = f"{type(exc).__name__}: {exc}"
        state = "INCOMPLETE"
    finally:
        if started is not None:
            elapsed_s = (clock() - started) / 1e9
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async def close(node_id):
            try:
                await asyncio.wait_for(workers[node_id].close(), close_timeout_s)
            except Exception as exc:
                errors[f"close:{node_id}"] = f"{type(exc).__name__}: {exc}"
        await asyncio.gather(*(close(node) for node in nodes))
    report = (coordinated_summary(summaries, expected_nodes=nodes, elapsed_s=elapsed_s)
              if elapsed_s > 0 else {"status": "INCOMPLETE", "coordinated_fps": None,
                                    "missing_nodes": nodes, "nodes": summaries})
    report.update(lifecycle_state=state, errors=errors, configured_nodes=nodes)
    if errors:
        report["status"] = "INCOMPLETE"
    return report
