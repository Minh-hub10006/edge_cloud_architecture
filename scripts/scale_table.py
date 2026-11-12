"""TA@D vs number of Edges sharing one Cloud GPU: fixed models, heuristics, static splits and DRL actors.

  python -B scripts/scale_table.py TABLE.jsonl SCENE.jsonl [--actors a.json ...] [--edges 4 6 8 10 12 16 20]

Scenario per N: N Edges (real profiles with jitter + synthetic devices), LAN, 70-100% of the GPU, test split.
Static split p: a fixed share p of the Edges (drawn per episode) run RT-DETR on the Cloud, the rest YOLO11s.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl import mappo  # noqa: E402
from drl.sim import MultiEdgeSim, QualityTable, SimConfig  # noqa: E402
from edge_agent.drl_policy import ACTION_MODES, MlpActor  # noqa: E402

RT, Y11 = ACTION_MODES.index("rtdetr_cloud"), ACTION_MODES.index("yolo11s_edge")


def split_ta(table, cfg, p, deadline, episodes, seed=1234):
    sim, rng, ta = MultiEdgeSim(table, cfg, seed=seed), random.Random(7), []
    for _ in range(episodes):
        sim.reset()
        n = len(sim.edges)
        cloud = set(rng.sample(range(n), round(p * n)))
        done = False
        while not done:
            masks = sim.masks()
            _o, _r, done, info = sim.step([RT if i in cloud and masks[i][RT] else Y11 for i in range(n)])
            ta.extend(e["ta"][deadline] for e in info["edges"])
    return float(np.mean(ta))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("table")
    p.add_argument("scene")
    p.add_argument("--actors", nargs="*", default=[])
    p.add_argument("--edges", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16, 20])
    p.add_argument("--deadline", type=int, default=200)
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args(argv)
    table = QualityTable.load([args.table], args.scene, split="test")
    actors = {Path(a).stem: MlpActor.load(a) for a in args.actors}
    cols = ["always_yolo11s_edge", "always_rtdetr_cloud", "always_lwdetr_large_cloud", "rules_acc"]
    names = ["YOLO11s", "RT-DETR", "LW-DETR", "heuristic", "split 25%", "split 50%", "cap K=8", "cap K=10", *actors]
    print(f"TA@{args.deadline} by number of Edges\n")
    print("| Edges | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    key = f"ta_{args.deadline}"
    for n in args.edges:
        cfg = SimConfig(a_min=.75, **dict(mappo.SCALE8, edges=(n, n)))
        cells = [mappo.evaluate(table, policy=c, cfg=cfg, episodes=args.episodes)[key] for c in cols]
        cells += [split_ta(table, cfg, share, args.deadline, args.episodes) for share in (.25, .5)]
        # Capacity-aware split: at most K Cloud streams (measured GPU capacity ~ K x 15 FPS), the rest local.
        cells += [split_ta(table, cfg, min(1.0, k / n), args.deadline, args.episodes) for k in (8, 10)]
        cells += [mappo.evaluate(table, a, policy="actor", cfg=cfg, episodes=args.episodes)[key] for a in actors.values()]
        best = max(cells)
        print(f"| {n} | " + " | ".join(f"**{c:.3f}**" if c == best else f"{c:.3f}" for c in cells) + " |", flush=True)


if __name__ == "__main__":
    main()
