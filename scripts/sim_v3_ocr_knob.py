"""Round 22 gap test (30/10): is there causal headroom in controlling the OCR side (workers over time)?

Detection placement is held fixed (the live `rules`: 4 streams D-FINE-X@15, or every stream on RT-DETR@15) so only the
OCR knob differs. Knob policies: fixed 2/4/6/8 workers; `now` (causal: the OCR service's queue and arrival rate this
second -> the fewest workers that keep up, more while a queue is waiting); `oracle` (each decision tries 2/4/6/8 on a
copy of the sim for the next 5 s and keeps the best: an upper bound). OCR must keep up: longest wait <= 60 s and the
queue left at the end <= 30 s of reading. Metric: constrained FPS (F1 >= .70, deadline 150 ms), TA@150.

  PYTHONPATH=. python -B scripts/sim_v3_ocr_knob.py
"""
import copy
import dataclasses
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import SimV3, TableV3, metrics, ocr_busy  # noqa: E402
from drl.train_v3 import config  # noqa: E402
from scripts.sim_v3_baselines import DX, PROBES, RT, TABLE, rules  # noqa: E402

WORKERS = (2, 4, 6, 8)
NS, EPISODES = (10, 14), 10
CELLS = [(r, vn, dyn) for r in (.45, .7) for vn in (.4, .7) for dyn in ("none", "all")]
DETECT = {"rules": rules(4, DX, 15), "allrt": rules(1.0, RT, 15)}
MAX_WAIT_S, END_WAIT_S = 60.0, 30.0
_G = {}


def _init():
    _G["test"] = TableV3.load(TABLE, "test", probe_paths=PROBES)


def cfg_for(reads, vn, dyn):
    return dataclasses.replace(
        config("probe", "test", False, scenarios=None if dyn == "none" else ["surge", "diurnal", "edgeslow", "thermal"],
               deadline_ms=150.0),
        ocr_read_ms=50.0, ocr_crops_per_vehicle=reads, ocr_ms_per_crop=0.0, vn_share=vn, dynamics_keep_dataset=True,
        ocr_backlog=True, ocr_workers=4)


def wait_s(sim, reads_now=None):
    """Seconds to read the queue at full speed with the current workers."""
    cap = sim.ocr_workers * 1000 / ocr_busy(1e9, sim.ocr_workers, sim.cfg)[1]
    return sim.ocr_queue / cap


def score(secs, f1_min=.70):
    """Soft constrained FPS of a stretch of seconds (as the training reward), for the oracle's choice."""
    n = len(secs[0])
    out = 0.0
    for i in range(n):
        eff = np.mean([s[i]["fps"] * s[i]["done"] * s[i]["on_time"] for s in secs])
        f1 = np.mean([s[i]["f1"] for s in secs])
        drop = 1 - np.mean([s[i]["done"] * s[i]["on_time"] for s in secs])
        out += eff * np.clip((f1 - (f1_min - .05)) / .10, 0, 1) * np.clip((.10 - drop) / .05, 0, 1)
    return out


def knob_now(sim):
    """Causal: arrival rate and queue of the last second (OCR service stats) -> workers."""
    reads = sim.ocr_crops * sum(s.cell().get("new", 0) for s in sim.streams)
    if wait_s(sim) > 10:
        return max(WORKERS)
    for k in WORKERS:  # fewest workers whose full-speed capacity covers the arrivals with 15 % slack
        if k * 1000 / ocr_busy(1e9, k, sim.cfg)[1] >= 1.15 * reads:
            return k
    return max(WORKERS)


def knob_oracle(sim, acts):
    best = None
    for k in WORKERS:
        trial = copy.deepcopy(sim, memo={id(sim.table): sim.table})
        trial.ocr_workers = k
        secs = trial.step(acts)
        ok = wait_s(trial) <= MAX_WAIT_S / 2  # keep room for later surges
        val = score(secs) - (0 if ok else 1e3)
        if best is None or val > best[0]:
            best = (val, k)
    return best[1]


def episode(cfg, n, seed, detect, knob):
    sim = SimV3(_G["test"], cfg, seed)
    sim.reset(n)
    secs, max_wait = [], 0.0
    for _ in range(cfg.episode_decisions):
        acts = DETECT[detect](sim)
        sim.ocr_workers = (knob if isinstance(knob, int) else knob_now(sim) if knob == "now" else knob_oracle(sim, acts))
        out = sim.step(acts)
        secs += out
        max_wait = max(max_wait, wait_s(sim))
    m = metrics(secs, f1_target=.70)
    return {"c70": m["fps_goodput_c70"], "ta": m["ta200"], "max_wait": max_wait, "end_wait": wait_s(sim),
            "workers": float(np.mean([s[0]["ocr_workers"] for s in secs]))}


def task(args):
    (reads, vn, dyn), n, seed, detect = args
    cfg = cfg_for(reads, vn, dyn)
    return (reads, vn, dyn, n, detect), {k: episode(cfg, n, seed, detect, k) for k in list(WORKERS) + ["now", "oracle"]}


def main():
    jobs = [(c, n, s, d) for c in CELLS for n in NS for s in range(EPISODES) for d in DETECT]
    with mp.Pool(14, initializer=_init) as pool:
        rows = pool.map(task, jobs, chunksize=1)
    print("| reads/vehicle | VN | dynamics | N | detection | best fixed workers (keeps up) | c70 fixed | c70 now | "
          "c70 oracle | now vs fixed | oracle vs fixed | now max/end wait s | fixed workers that keep up |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted({r[0] for r in rows}):
        eps = [r[1] for r in rows if r[0] == key]
        agg = {k: {f: float(np.mean([e[k][f] for e in eps])) for f in eps[0][k]} for k in eps[0]}
        keeps = [k for k in WORKERS if agg[k]["max_wait"] <= MAX_WAIT_S and agg[k]["end_wait"] <= END_WAIT_S]
        if not keeps:
            print(f"| {' | '.join(map(str, key))} | none keeps up | - | {agg['now']['c70']:.1f} | {agg['oracle']['c70']:.1f} "
                  f"| - | - | {agg['now']['max_wait']:.0f}/{agg['now']['end_wait']:.0f} | - |")
            continue
        fk = max(keeps, key=lambda k: agg[k]["c70"])
        f = agg[fk]["c70"]
        print(f"| {' | '.join(map(str, key))} | {fk} | {f:.1f} | {agg['now']['c70']:.1f} | {agg['oracle']['c70']:.1f} "
              f"| {(agg['now']['c70'] / max(f, 1e-9) - 1) * 100:+.1f}% | {(agg['oracle']['c70'] / max(f, 1e-9) - 1) * 100:+.1f}% "
              f"| {agg['now']['max_wait']:.0f}/{agg['now']['end_wait']:.0f} | {keeps} |", flush=True)


if __name__ == "__main__":
    main()
