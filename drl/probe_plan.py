"""Selective probing rule (round 27, 30/10; the rule of the round-26 planner): which streams send probe frames to
D-FINE-X in the next decision, for any scheduler (DRL, heuristics). Budget = one probe-second per stream not on
D-FINE-X (what the fixed scheme spends); each probe-second goes to the stream whose Edge F1 estimate is nearest the
.70 floor, whose probe is oldest or noisiest, with diminishing returns per stream (at most decision_s seconds).
The Edge F1 estimate is linear in the stream's weighted probe history (SimV3 _probe_history), fitted on train."""
from __future__ import annotations

import numpy as np

_PROFILES = {}


def edge_profile(table, action=("yolo11s_edge", 15)):
    """(c0, c1): F1 of the Edge model ~ c0 + c1 x probe over every train second with a probe (cached per table)."""
    key = (id(table), action)
    if key not in _PROFILES:
        cells = [c for v in table.seq.values() for c in v if c.get("probe") is not None]
        x = np.array([float(np.mean(c["probe"])) for c in cells])
        y = np.array([c[action] for c in cells])
        c1 = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        _PROFILES[key] = (float(y.mean() - c1 * x.mean()), c1)
    return _PROFILES[key]


def plan(sim, actions, profile, f1_floor=.70):
    """actions: the (mode, fps) chosen for the next decision -> {stream index: probe seconds}."""
    c0, c1 = profile
    pri = {}
    for i, (s, (mode, _)) in enumerate(zip(sim.streams, actions)):
        if mode == "dfine_x_cloud":
            continue  # its D-FINE-X output already shows how hard the scene is
        if s.probe_ema is None:
            pri[i] = 3.0  # never probed
            continue
        pri[i] = (np.exp(-abs(c0 + c1 * s.probe_ema - f1_floor) / .05) + min(1.0, (s.probe_age or 0) / 30)
                  + min(1.0, np.sqrt(s.probe_var) / 5))
    out, budget = {}, len(pri)
    for _ in range(budget):
        if not pri:
            break
        i = max(pri, key=lambda j: pri[j] / (1 + out.get(j, 0)))
        out[i] = out.get(i, 0) + 1
        if out[i] >= sim.cfg.decision_s:
            pri.pop(i)
    return out
