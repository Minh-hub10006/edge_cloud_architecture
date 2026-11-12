"""Other readings of the same simulated runs (test split, N = 4..12 pooled): error rate 1 - TA@200, share of
5-s camera windows below A_min (TA < .75), TA@200 on hard scenes only (night + rain), and the share of the
YOLO11s-to-ceiling gap closed (ceiling = LW-DETR@15 test F1 with every frame on time).

  python -B scripts/alt_metrics.py --central results/drl/b1/cb_s1.json [--episodes 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drl import central, mappo  # noqa: E402
from drl.sim import QualityTable, SimConfig  # noqa: E402

BASE = {"YOLO11s (Edge)": "always_yolo11s_edge", "RT-DETR (Cloud)": "always_rtdetr_cloud",
        "LW-DETR (Cloud)": "always_lwdetr_large_cloud", "Heuristic": "rules_fps"}
FLAGS = dict(coord_features=True, async_share=.34, fps_levels=(5, 10, 15))
CEILING = .889  # LW-DETR@15 mean test F1 (results/detrac/table_fps.jsonl)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--central", required=True)
    p.add_argument("--episodes", type=int, default=6)
    args = p.parse_args(argv)
    table = QualityTable.load([ROOT / "results/detrac/table_fps.jsonl"], ROOT / "results/detrac/scene.jsonl",
                              split="test")
    actor = central.CentralActor.load(args.central)
    for setting, scen in (("testbed", mappo.N_TESTBED), ("diverse", mappo.N_DIVERSE)):
        parts = {name: [] for name in [*BASE, "DRL"]}
        for n, kw in scen.items():
            cfg = SimConfig(a_min=.75, **kw, **FLAGS)
            for name, pol in BASE.items():
                parts[name] += mappo.evaluate(table, policy=pol, cfg=cfg, episodes=args.episodes, raw=True)
            parts["DRL"] += central.evaluate_central(table, actor, cfg, episodes=args.episodes)["parts"]
        floor = np.mean([q["ta"][200] for q in parts["YOLO11s (Edge)"]])
        print(f"\n{setting} (N 4-12 pooled)\n")
        print("| policy | TA@200 | error 1-TA | error vs YOLO11s | windows below .75 | TA@200 night+rain | gap closed |")
        print("|---|---|---|---|---|---|---|")
        for name, ps in parts.items():
            ta = np.array([q["ta"][200] for q in ps])
            hard = np.array([q["ta"][200] for q in ps if q.get("weather") in ("night", "rainy")])
            err, err0 = 1 - ta.mean(), 1 - floor
            print(f"| {name} | {ta.mean():.3f} | {err:.3f} | {100 * (err - err0) / err0:+.0f}% | "
                  f"{100 * (ta < .75).mean():.0f}% | {hard.mean():.3f} | "
                  f"{100 * (ta.mean() - floor) / (CEILING - floor):+.0f}% |", flush=True)


if __name__ == "__main__":
    main()
