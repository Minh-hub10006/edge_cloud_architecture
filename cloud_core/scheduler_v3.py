"""Cloud scheduler (docs/PLAN_CONTROL_V3_VI.md): every period, one joint decision -> a (model, FPS) command per stream.

Edges send a compact window state (SubmitWindowState); the registry keeps the latest per stream. The policy sees all
live streams plus the Cloud telemetry and returns per-stream actions; commands go back to each stream with its
feedback. Every decision is logged (state, commands) as the DRL dataset.
"""
from __future__ import annotations

import asyncio
import json
from time import monotonic_ns, time_ns

from contracts.control_v1 import encode, validate_window_state


class StreamRegistry:
    """Latest window state per stream; a stream silent for `expire_s` is dropped (stopped or crashed)."""

    def __init__(self, *, expire_s=15.0, clock=monotonic_ns):
        self.expire_ns, self.clock = round(expire_s * 1e9), clock
        self.latest = {}

    def update(self, value):
        value = validate_window_state(value)
        key = tuple(value["stream_key"])
        old = self.latest.get(key)
        if old is None or value["window_id"] >= old[0]["window_id"]:  # late duplicates never go back in time
            self.latest[key] = (value, self.clock())
        return key

    def live(self):
        now = self.clock()
        for key in [k for k, (_v, seen) in self.latest.items() if now - seen > self.expire_ns]:
            del self.latest[key]
        return {key: value for key, (value, _seen) in sorted(self.latest.items())}


class Scheduler:
    def __init__(self, registry, telemetry, policy, *, period_s=5.0, log=None, clock=monotonic_ns):
        if period_s <= 0:
            raise ValueError("scheduler period must be positive")
        self.registry, self.telemetry, self.policy = registry, telemetry, policy
        self.period_s, self.log, self.clock = period_s, log, clock
        self.sequence = 0
        self.commands = {}

    def command_for(self, stream_key):
        command = self.commands.get(tuple(stream_key))
        return command if command is not None and command["valid_until_utc_ns"] > time_ns() else None

    def step(self):
        streams = self.registry.live()
        if not streams:
            return {}
        cloud = self.telemetry.snapshot() if self.telemetry is not None else None
        actions = self.policy.decide(streams, cloud)
        self.sequence += 1
        now = time_ns()
        valid_until = now + round(3 * self.period_s * 1e9)  # survives two missed periods, then the Edge falls back
        commands = {}
        for key, (mode, fps) in actions.items():
            if key not in streams:
                continue
            commands[key] = {"model_mode": mode, "fps": float(fps), "valid_until_utc_ns": valid_until,
                             "decision_id": f"cloud/{self.sequence}/{':'.join(map(str, key))}"}
        self.commands = commands
        if self.log is not None:
            self.log.write(encode({"sequence": self.sequence, "decided_utc_ns": now, "policy": self.policy.name,
                                   "cloud": cloud, "streams": {":".join(map(str, k)): v for k, v in streams.items()},
                                   "commands": {":".join(map(str, k)): c for k, c in commands.items()}}).decode() + "\n")
            self.log.flush()
        return commands

    async def run(self):
        while True:
            try:
                self.step()
            except Exception as exc:  # a policy bug must not kill the receiver; Edges fall back on expiry
                print(f"[Scheduler] {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(self.period_s)


def load_policy(spec, rules_capacity_rps=85.0):
    """CLOUD_SCHEDULER: 'fixed:<json file>' | 'rules' | 'drl:<actor>' (DRL v3, PLAN_PROBE / PLAN_DRL_V3)."""
    from cloud_core.policies_v3 import FixedPolicy, RulesPolicy
    kind, _, arg = spec.partition(":")
    if kind == "fixed":
        with open(arg, encoding="utf-8") as file:
            return FixedPolicy(json.load(file))
    if kind == "rules":
        return RulesPolicy(rules_capacity_rps)
    if kind == "drl":
        raise ValueError("drl scheduler policy arrives with DRL v3 (actor format not final)")
    raise ValueError(f"unknown CLOUD_SCHEDULER {spec!r}")
