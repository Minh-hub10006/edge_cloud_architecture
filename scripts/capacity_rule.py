"""Capacity rule: pick (Cloud model, analytics FPS) and how many Edges go to the Cloud from the GPU capacity alone.

For N Edges, GPU capacity C_m frames/s for model m and safety share rho:
  1. if some (m, f) fits every Edge on the Cloud (N x f <= rho x C_m), take the most accurate such pair;
  2. otherwise put k = floor(rho x C_m / f) Edges on (m, f) and the rest on YOLO11s at 15 FPS,
     choosing the (m, f, k) with the best mean expected F1.
Expected F1 per (model, FPS) comes from the TRAIN split only; evaluation runs on the test split.

  python -B scripts/capacity_rule.py [--rho .7 .85] [--episodes 12]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from drl.sim import CLOUD_MODES, MultiEdgeSim, QualityTable, SimConfig  # noqa: E402
from fps_eval import FPS, SCEN  # noqa: E402

LOCAL = ("yolo11s_edge", 15)


def expected_f1(table):
    """Mean F1 per (mode, FPS) over a table's cells."""
    out = {}
    for mode in ("yolo11s_edge", "rtdetr_cloud", "lwdetr_large_cloud"):
        for f in FPS:
            out[(mode, f)] = float(np.mean([table.f1(k, mode, f) for k in table.cells]))
    return out


def plan(n, capacity, f1, rho):
    """-> (cloud (mode, fps), number of Edges on the Cloud) maximizing the mean expected F1."""
    best = (f1[LOCAL], None, 0)
    for mode in CLOUD_MODES:
        for f in FPS:
            k = min(n, int(rho * capacity[mode] // f))
            value = (k * f1[(mode, f)] + (n - k) * f1[LOCAL]) / n
            if value > best[0]:
                best = (value, (mode, f), k)
    return best[1], best[2]


def run(table, kw, f1, rho, episodes):
    cfg = SimConfig(a_min=.75, **kw, fps_levels=FPS)
    sim, rng, ta, chosen = MultiEdgeSim(table, cfg, seed=1234), random.Random(7), [], set()
    for _ in range(episodes):
        sim.reset()
        n = len(sim.edges)
        pair, k = plan(n, cfg.cloud_capacity, f1, rho)
        chosen.add(f"{pair[0].split('_')[0]}@{pair[1]} x{k}/{n}" if pair else f"YOLO11s x{n}")
        cloud = set(rng.sample(range(n), k))
        done = False
        while not done:
            acts = []
            for i, m in enumerate(sim.masks()):
                want = sim.action_index(*pair) if (pair and i in cloud) else sim.action_index(*LOCAL)
                acts.append(want if m[want] else sim.action_index(*LOCAL))
            _o, _r, done, info = sim.step(acts)
            ta.extend(e["ta"][200] for e in info["edges"])
    return float(np.mean(ta)), sorted(chosen)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rho", type=float, nargs="+", default=[.7, .85])
    p.add_argument("--episodes", type=int, default=12)
    args = p.parse_args(argv)
    paths = ([ROOT / "results/detrac/table_fps.jsonl"], ROOT / "results/detrac/scene.jsonl")
    f1 = expected_f1(QualityTable.load(*paths, split="train"))
    print("expected F1 (train):", {f"{m.split('_')[0]}@{f}": round(v, 3) for (m, f), v in f1.items()})
    test = QualityTable.load(*paths, split="test")
    print("\n| rule | " + " | ".join(SCEN) + " |\n|---|" + "---|" * len(SCEN))
    for rho in args.rho:
        cells, plans = [], []
        for kw in SCEN.values():
            value, chosen = run(test, kw, f1, rho, args.episodes)
            cells.append(f"{value:.3f}")
            plans.append(", ".join(chosen))
        print(f"| capacity rule rho={rho} | " + " | ".join(cells) + " |")
        print(f"| (chosen) | " + " | ".join(plans) + " |", flush=True)


if __name__ == "__main__":
    main()
