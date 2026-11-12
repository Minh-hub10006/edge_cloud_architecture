"""Round 30 (1/11): the hand rule ("k cameras on Cloud", sim_v3_baselines.rules) tuned on TRAIN for a grid of
situations (OCR reads per vehicle x OCR workers x VN share; fixed cameras), exactly as the evaluation tunes its
baseline (sim_v3_oracle_shared.tune). DRL round 30 imitates the rule of the nearest grid point before PPO, so it
starts from the baseline instead of the planner.

  PYTHONPATH=. python -B scripts/rule_table.py results/drl_r30/rule_table.json
"""
import itertools
import json
import multiprocessing as mp
import sys

import numpy as np

GRID = list(itertools.product((0.0, .45, .9), (4, 8), (.4, .55, .7)))


def lookup(table, reads, workers, vn):
    """The tuned rule of the grid point nearest to this situation."""
    return min(table, key=lambda r: abs(r["reads"] - reads) / .45 + abs(r["workers"] - workers) / 4
               + abs(r["vn"] - vn) / .15)["rule"]


def _fe():
    if "--fixed-camera" not in sys.argv:
        sys.argv.append("--fixed-camera")
    import scripts.fair_eval as fe
    return fe


def score(args):
    d, rule = args
    o = _fe().o
    cfg = o.cfg_for("train", d, "none")
    return d, rule, float(np.mean([o.episode(o._G["train"], cfg, n, s, o.rules(*rule))[o.TUNE_KEY]
                                   for n in (6, 10, 14) for s in range(3)]))


def main(out):
    fe = _fe()
    o = fe.o
    rules_grid = [(x / 10, m, f) for x in range(0, 11, 2) for m in (o.DX, o.RT) for f in (15, 10)]
    with mp.Pool(14, initializer=fe.cb._init) as pool:
        res = pool.map(score, [(d, r) for d in GRID for r in rules_grid], chunksize=4)
    table = []
    for d in GRID:
        best = max((x for x in res if x[0] == d), key=lambda x: x[2])
        table.append({"reads": d[0], "workers": d[1], "vn": d[2], "rule": list(best[1]), "c70": round(best[2], 1)})
        print(table[-1])
    assert lookup(table, .45, 8, .7) == table[GRID.index((.45, 8, .7))]["rule"]
    json.dump(table, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
