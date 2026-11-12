"""Central allocation (round 21, 30/10): per-stream action scores -> one joint assignment that never overfills the
Cloud. Every stream still gets its own (model, FPS); the allocator only decides which streams get Cloud room.

Streams are ranked by how much their best Cloud action scores above their best Edge action; in that order a stream gets
its best-scoring Cloud action whose predicted GPU load stays <= gpu_cap and predicted Cloud latency <= latency_share x
the deadline, else its best Edge action. Capacity and latency come from what the Cloud scheduler can see: nominal model
capacity, the GPU monitor's free share, and the OCR service's load over the last second (drl/sim_v3 live OCR model).
"""
from __future__ import annotations

import numpy as np

from drl.sim import _md1_ms
from drl.sim_v3 import CLOUD


def cloud_capacity(sim):
    """{model: images/s}, {model: extra ms per call from OCR} for this decision."""
    cfg = sim.cfg
    base = {m: c * cfg.gpu_efficiency * sim.bg_base * sim.thermal_factor() for m, c in cfg.capacity.items()}
    if cfg.ocr_read_ms:
        slow = 1 / (1 - min(.9, sim.downstream))
        return ({m: c / slow ** cfg.ocr_cap_exponent for m, c in base.items()},
                {m: (slow - 1) * cfg.cloud_inflight * 1000 / c for m, c in base.items()})
    slow = 1 + cfg.det_stretch_per_ocr_share * sim.downstream
    return {m: c / slow for m, c in base.items()}, dict.fromkeys(base, 0.0)


def _fits(sim, cloud, caps, stretch, gpu_cap, on_time_q):
    """cloud: [(stream, mode, fps)] -> True if the GPU load stays <= gpu_cap and every one of them still meets the
    deadline at quantile on_time_q. One FIFO for every Cloud model (as drl/sim_v3): a long stretched D-FINE call
    delays the RT-DETR calls behind it."""
    caps_load = sum(f / caps[m] for _, m, f in cloud) + sim._probe_load(caps)
    if caps_load > gpu_cap:
        return False
    svc = {m: 1000 / caps[m] + stretch[m] for _, m, _ in cloud}
    mean_svc = sum(f * svc[m] for _, m, f in cloud) / sum(f for _, _, f in cloud)
    wait = _md1_ms(min(caps_load, .95) * 1000 / mean_svc, 1000 / mean_svc)
    # latency is lognormal (sigma latency_sigma) around the model: the mean must leave room for the on_time_q tail
    limit = sim.cfg.deadline_ms / np.exp(on_time_q * sim.cfg.latency_sigma)
    return all(sim.streams[i].edge.join_ms + svc[m] + wait <= limit for i, m, _ in cloud)


def allocate(sim, scores, actions, gpu_cap=.9, on_time_z=1.645):
    """scores: [streams, len(actions)] (higher = better); actions: [(mode, fps)] -> one (mode, fps) per stream.
    on_time_z: normal quantile of the share of frames that must meet the deadline (1.645 = 95 %, the c70 drop cap)."""
    scores = np.asarray(scores, dtype=float)
    caps, stretch = cloud_capacity(sim)
    cloud_idx = [k for k, (m, _) in enumerate(actions) if m in CLOUD]
    edge_idx = [k for k, (m, _) in enumerate(actions) if m not in CLOUD]
    best_edge = [max(edge_idx, key=lambda k: row[k]) for row in scores]
    gain = [max(row[k] for k in cloud_idx) - row[e] for row, e in zip(scores, best_edge)]
    chosen, cloud = list(best_edge), []
    for i in sorted(range(len(scores)), key=lambda i: -gain[i]):
        if gain[i] <= 0:
            break
        for k in sorted(cloud_idx, key=lambda k: -scores[i][k]):
            if scores[i][k] <= scores[i][best_edge[i]]:
                break
            mode, fps = actions[k]
            if _fits(sim, cloud + [(i, mode, fps)], caps, stretch, gpu_cap, on_time_z):
                chosen[i] = k
                cloud.append((i, mode, fps))
                break
    return [actions[k] for k in chosen]


def profile_scores(sim, profile, actions, f1_min=.70):
    """Model-based scores (round 21 MPC baseline): predicted F1 from the probe (offline profile, as JCAB) -> the
    constrained-FPS value of each action: FPS/15 while the predicted F1 clears f1_min (soft edge +-.05, as the reward)."""
    out = []
    for s in sim.streams:
        x = s.probe_last if s.probe_last is not None else 0.0
        row = []
        for a in actions:
            c0, c1 = profile[a]
            row.append(a[1] / 15 * float(np.clip((c0 + c1 * x - (f1_min - .05)) / .10, 0, 1)) + 1e-3 * a[1])
        out.append(row)
    return np.array(out)
