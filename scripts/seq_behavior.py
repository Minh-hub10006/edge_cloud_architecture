"""What the sequential Cloud scheduler does: share of camera-decisions on the Cloud (and at which FPS) by scene
regime (weather) and by camera count, test split, diverse scenario (scene regimes on).

  python -B scripts/seq_behavior.py ACTOR.json [--episodes 6]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drl import mappo, seq  # noqa: E402
from drl.sim import CLOUD_MODES, QualityTable, SimConfig  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("actor")
    p.add_argument("--episodes", type=int, default=6)
    args = p.parse_args(argv)
    table = QualityTable.load([ROOT / "results/detrac/table_fps.jsonl"], ROOT / "results/detrac/scene.jsonl",
                              split="test")
    actor = seq.SeqActor.load(args.actor)
    by_weather, by_n = defaultdict(list), {}
    for n, kw in mappo.N_DIVERSE.items():
        cfg = SimConfig(a_min=.75, **kw, fps_levels=(5, 10, 15))
        sim_parts = seq.evaluate_seq(table, actor, cfg, episodes=args.episodes)
        parts = sim_parts["parts"]
        by_n[n] = (sim_parts["cloud_share"], sim_parts["ta_200"],
                   {m: float(np.mean([q["mode"] == m for q in parts])) for m in
                    ("yolo_edge", "yolo11s_edge", "rtdetr_cloud", "lwdetr_large_cloud")},
                   {f: float(np.mean([q["fps"] == f for q in parts])) for f in (5, 10, 15)})
    # weather: rerun one scenario tracking each camera's regime
    cfg = SimConfig(a_min=.75, **mappo.N_DIVERSE[8], fps_levels=(5, 10, 15))
    sim = seq.MultiEdgeSim(table, seq._cfg(cfg), seed=99)
    import random
    rng = random.Random(1)
    for _ in range(args.episodes):
        rounds, parts = seq.run_episode(sim, actor, rng=rng)
        # parts are per decision x camera in camera order; regimes stay fixed (switch_p tiny) within an episode
        n = len(sim.edges)
        for k, q in enumerate(parts):
            by_weather[sim.edges[k % n].regime].append(q["mode"] in CLOUD_MODES)
    print("| cameras | Cloud share | TA@200 | modes Y8/Y11/RT/LW | FPS 5/10/15 |\n|---|---|---|---|---|")
    for n, (share, ta, modes, fps) in by_n.items():
        print(f"| {n} | {share:.2f} | {ta:.3f} | " + "/".join(f"{v:.2f}" for v in modes.values()) + " | "
              + "/".join(f"{v:.2f}" for v in fps.values()) + " |")
    print("\n| scene regime (8 cameras) | Cloud share | decisions |\n|---|---|---|")
    for w, xs in sorted(by_weather.items()):
        print(f"| {w} | {np.mean(xs):.2f} | {len(xs)} |")


if __name__ == "__main__":
    main()
