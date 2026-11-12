"""TA@200 of model+FPS DRL actors vs standard baselines (each model at 15 FPS, heuristics) and fixed
(model, FPS) pairs, from 3 to 20 Edges and under within-episode dynamics (simulator, test split).

  python -B scripts/fps_eval.py RUN_DIR GROUP [GROUP ...] [--episodes 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drl import mappo  # noqa: E402
from drl.sim import MultiEdgeSim, QualityTable, SimConfig  # noqa: E402

SCEN = {"3 real": dict(real_edges_only=True, network_ms=(3, 3), wan_p=0, cloud_background=(1, 1)),
        "8": mappo.SCALE8, "12": mappo.SCALE12, "16": mappo.SCALE16, "20": dict(mappo.SCALE8, edges=(20, 20)),
        "12 dyn": mappo.DYN12}
STANDARD = {"YOLOv8n": "always_yolo_edge", "YOLO11s": "always_yolo11s_edge", "RT-DETR": "always_rtdetr_cloud",
            "LW-DETR": "always_lwdetr_large_cloud", "heuristic": "rules", "heuristic-acc": "rules_acc"}
FIXED = (("lwdetr_large_cloud", 10), ("lwdetr_large_cloud", 5), ("rtdetr_cloud", 10), ("yolo11s_edge", 10))
FPS = (5, 10, 15)


def fixed_ta(table, kw, mode, fps, episodes):
    sim = MultiEdgeSim(table, SimConfig(a_min=.75, **kw, fps_levels=FPS), seed=1234)
    ta = []
    for _ in range(episodes):
        sim.reset()
        done = False
        while not done:
            want = sim.action_index(mode, fps)
            fallback = sim.action_index("yolo_edge", fps)
            _o, _r, done, info = sim.step([want if m[want] else fallback for m in sim.masks()])
            ta.extend(e["ta"][200] for e in info["edges"])
    return float(np.mean(ta))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("groups", nargs="+")
    p.add_argument("--episodes", type=int, default=12)
    args = p.parse_args(argv)
    table = QualityTable.load([ROOT / "results/detrac/table_fps.jsonl"], ROOT / "results/detrac/scene.jsonl",
                              split="test")
    d = Path(args.run_dir)
    val = {}
    for log in d.glob("train_*.log"):
        ev = [json.loads(x)["eval_team_return"] for x in log.read_text().splitlines() if "eval_team_return" in x]
        val[log.stem.replace("train_", "")] = max(ev)
    best = {g: max((k for k in val if k.startswith(g + "_")), key=val.get) for g in args.groups}
    print("validation:", {k: round(v, 2) for k, v in sorted(val.items())}, "best:", best, flush=True)
    actors = {k: mappo.load_actor(d / f"actor_{k}.json") for k in val}
    rows = {}
    for sc, kw in SCEN.items():
        r = {n: mappo.evaluate(table, policy=pol, cfg=SimConfig(a_min=.75, **kw), episodes=args.episodes)["ta_200"]
             for n, pol in STANDARD.items()}
        for mode, fps in FIXED:
            r[f"fixed {mode.split('_')[0]}@{fps}"] = fixed_ta(table, kw, mode, fps, args.episodes)
        flags = dict(coord_features=True, async_share=.34, fps_levels=FPS)
        per = {k: mappo.evaluate(table, a, policy="actor", cfg=SimConfig(a_min=.75, **kw, **flags),
                                 episodes=args.episodes) for k, a in actors.items()}
        for g in args.groups:
            xs = [v["ta_200"] for k, v in per.items() if k.startswith(g + "_")]
            r[f"DRL {g} (val best)"] = per[best[g]]["ta_200"]
            r[f"DRL {g} mean"] = float(np.mean(xs))
            r[f"DRL {g} std"] = float(np.std(xs))
        rows[sc] = r
        print(sc, "done", flush=True)
    print("\nTA@200 by number of Edges (test split)\n")
    print("| policy | " + " | ".join(SCEN) + " |\n|---|" + "---|" * len(SCEN))
    for n in next(iter(rows.values())):
        print(f"| {n} | " + " | ".join(f"{rows[sc][n]:.3f}" for sc in SCEN) + " |")


if __name__ == "__main__":
    main()
