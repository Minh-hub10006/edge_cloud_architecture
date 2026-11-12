"""Main evaluation table: TA@D of the four single models, the accuracy/FPS heuristic and DRL actors for 4-12
cameras (simulator calibrated on the live 4-12 stream runs, test split), in two settings: the measured testbed
and the testbed with external conditions (GPU tenants, Edge CPU contention, encoder swings, network spells).

  python -B scripts/n_table.py [--actors A.json ...] [--episodes 12] [--deadline 200] [--out table.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drl import central, mappo, seq  # noqa: E402
from drl.sim import QualityTable, SimConfig  # noqa: E402

ROWS = {"YOLOv8n (Edge)": "always_yolo_edge", "YOLO11s (Edge)": "always_yolo11s_edge",
        "RT-DETR (Cloud)": "always_rtdetr_cloud", "LW-DETR (Cloud)": "always_lwdetr_large_cloud",
        "Heuristic (accuracy/FPS)": "rules_fps"}
FLAGS = dict(coord_features=True, async_share=.34, fps_levels=(5, 10, 15))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--actors", nargs="*", default=[])
    p.add_argument("--central", nargs="*", default=[], help="centralized Cloud scheduler actors (drl.central)")
    p.add_argument("--seq", nargs="*", default=[], help="sequential Cloud scheduler actors (drl.seq)")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--deadline", type=int, default=200)
    p.add_argument("--settings", nargs="+", default=["testbed", "external"])
    p.add_argument("--out")
    args = p.parse_args(argv)
    table = QualityTable.load([ROOT / "results/detrac/table_fps.jsonl"], ROOT / "results/detrac/scene.jsonl",
                              split="test")
    key = f"ta_{args.deadline}"
    policies = {**{name: (pol, None) for name, pol in ROWS.items()},
                **{f"DRL {Path(a).stem}": ("actor", mappo.load_actor(a)) for a in args.actors},
                **{f"DRL central {Path(a).stem}": ("central", central.CentralActor.load(a)) for a in args.central},
                **{f"DRL seq {Path(a).stem}": ("seq", seq.SeqActor.load(a)) for a in args.seq}}
    scenarios = {"testbed": mappo.N_TESTBED, "external": mappo.N_EVAL, "diverse": mappo.N_DIVERSE}
    out = {}
    for setting in args.settings:
        scen, rows = scenarios[setting], {}
        for name, (pol, actor) in policies.items():
            rows[name] = {n: (central.evaluate_central(table, actor, SimConfig(a_min=.75, **kw, **FLAGS),
                                                       episodes=args.episodes, deadline=args.deadline)
                              if pol == "central" else
                              seq.evaluate_seq(table, actor, SimConfig(a_min=.75, **kw, **FLAGS),
                                               episodes=args.episodes, deadline=args.deadline)
                              if pol == "seq" else
                              mappo.evaluate(table, actor, policy=pol, cfg=SimConfig(a_min=.75, **kw, **FLAGS),
                                             episodes=args.episodes))[key] for n, kw in scen.items()}
            print(setting, name, {n: round(v, 3) for n, v in rows[name].items()}, flush=True)
        ns = list(scen)
        print(f"\nTA@{args.deadline} by number of cameras, {setting} (test split)\n")
        print("| policy | " + " | ".join(str(n) for n in ns) + " | mean |\n|---|" + "---|" * (len(ns) + 1))
        best = {n: max(r[n] for r in rows.values()) for n in ns}
        for name, r in rows.items():
            cells = [f"**{r[n]:.3f}**" if r[n] == best[n] else f"{r[n]:.3f}" for n in ns]
            print(f"| {name} | " + " | ".join(cells) + f" | {np.mean([r[n] for n in ns]):.3f} |", flush=True)
        out[setting] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
