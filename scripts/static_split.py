"""Strongest non-learning reference for the centralized scheduler: fixed (Cloud model, FPS, k) splits with the
same per-camera ranking. Reports, per setting, the best single split over all camera counts (what a tuned static
rule achieves) and the best split chosen separately for each N (static oracle that knows N in advance).

  python -B scripts/static_split.py [--rank benefit] [--episodes 6]
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

SPLITS = [(m, f, k) for m in central.MODELS for f in (10, 15) for k in range(2, 13)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rank", default="benefit")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--settings", nargs="+", default=["testbed", "diverse"])
    args = p.parse_args(argv)
    table = QualityTable.load([ROOT / "results/detrac/table_fps.jsonl"], ROOT / "results/detrac/scene.jsonl",
                              split="test")
    acts = central.central_actions((5, 10, 15))
    scen = {"testbed": mappo.N_TESTBED, "diverse": mappo.N_DIVERSE}
    for setting in args.settings:
        res = {}
        for n, kw in scen[setting].items():
            cfg = SimConfig(a_min=.75, **kw, fps_levels=(5, 10, 15))
            for split in SPLITS:
                a = acts.index((split[0], split[1], min(split[2], n)))
                res[(split, n)] = central.evaluate_central(table, lambda o, m, a=a: a if m[a] else 0, cfg,
                                                           episodes=args.episodes, rank=args.rank)["ta_200"]
        ns = list(scen[setting])
        single = max(SPLITS, key=lambda s: np.mean([res[(s, n)] for n in ns]))
        print(f"{setting}: best single split {single[0].split('_')[0]}@{single[1]} k={single[2]}: "
              + " ".join(f"{res[(single, n)]:.3f}" for n in ns) + f" mean {np.mean([res[(single, n)] for n in ns]):.3f}")
        per_n = {n: max(SPLITS, key=lambda s: res[(s, n)]) for n in ns}
        print(f"{setting}: best split per N: " + " ".join(
            f"N{n}={per_n[n][0].split('_')[0]}@{per_n[n][1]}k{per_n[n][2]}:{res[(per_n[n], n)]:.3f}" for n in ns)
            + f" mean {np.mean([res[(per_n[n], n)] for n in ns]):.3f}", flush=True)


if __name__ == "__main__":
    main()
