"""Headroom with a shared Cloud (plate OCR on the same GPU, load ~ new vehicles on every camera), LAN, N 6-14.

  PYTHONPATH=. python -B scripts/sim_v3_oracle_shared.py

Grid: OCR crops per vehicle {0, 1, 2, 3} at 3.85 GPU-ms/crop (4090 co-run; ~0/18/35/53% GPU at N=10) x dynamics {none,
all four: surge, diurnal, edgeslow, thermal}. Baselines tuned on train per cell: static rule (share x Cloud model x FPS),
threshold heuristic (theta and its GPU cap), plus the untuned net heuristic. Oracle (best in hindsight per test episode and metric): a
greedy that knows each stream's next 5 s F1 and the next 5 s downstream load (Cloud gains first while the predicted GPU
stays under a cap .6-1.0), and a rescue oracle (streams below F1 .6 on the Edge to the Cloud model) under the same caps.
Metrics: TA@200 and constrained FPS goodput (F1 >= .6, drop <= 5%).

--live (round 19): the live OCR model fitted on the testbed 30/10 (drl/sim_v3 ocr_read_ms): reads per vehicle
{0, .45 = testbed, .7} x OCR workers {4, 8} x VN share {.4, .7 = testbed} x dynamics; headline constrained FPS at
F1 >= .70 and deadline 150 ms (agreed 30/10; --d200 for the old deadline), also F1 .60 / .75 and TA; baselines tuned
on train for the headline metric; DRL actors results/drl_ocr2c70_d150/actor_probe_ocr2_fps3c70_d150_s*.pt.
"""
import dataclasses
import glob
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import CLOUD, SimV3, TableV3, metrics, ocr_busy  # noqa: E402
from drl.train_v3 import config  # noqa: E402
from scripts.sim_v3_baselines import (TABLE, PROBES, THETAS, HeuristicThreshold, JCAB, heuristic_net, rules,  # noqa: E402
                                      Y11, DX, RT)

NS, EPISODES = (6, 8, 10, 12, 14), 10
OCR_MS = 3.85  # GPU-busy ms per EasyOCR call, 4090 co-run measurement 29/10 (scripts/corun_measure.py)
DOWN = (0, 1, 2, 3)  # crops OCRed per vehicle
DYN = {"none": None, "all": ["surge", "diurnal", "edgeslow", "thermal"]}
CAPS = (.6, .7, .8, .9, 1.0)
KEYS = ("ta200", "fps_goodput_c60")
ACTORS = sorted(glob.glob("results/drl_ocr/actor_probe_ocr_fps3_s*.pt"))  # round 18 DRL, if trained
LIVE = "--live" in sys.argv
TUNE_KEY = "ta200"
if LIVE:  # (reads per vehicle, OCR workers, VN share); no OCR: one workers value
    DOWN = tuple((d, w, vn) for vn in (.4, .7) for d, w in ((0.0, 4), (.45, 4), (.45, 8), (.7, 4), (.7, 8)))
    # round 20 (BC warm start, N 6-14, VN share .4-.7); round 19 actors: results/drl_ocr2c70_d150
    ACTORS = sorted(glob.glob("results/drl_r20/actor_probe_ocr2_fps3c70_d150_s*.pt"))
    KEYS = ("fps_goodput_c70", "fps_goodput_c60", "fps_goodput_c75", "ta200")
    TUNE_KEY = "fps_goodput_c70"
DEADLINE = 200.0 if not LIVE or "--d200" in sys.argv else 150.0
_G = {}


def ocr_kwargs(d):
    if LIVE:
        return {"ocr_read_ms": 50.0 if d[0] else 0.0, "ocr_crops_per_vehicle": d[0], "ocr_workers": d[1],
                "ocr_ms_per_crop": 0.0, "vn_share": d[2], "dynamics_keep_dataset": True}
    return {"ocr_ms_per_crop": OCR_MS if d else 0.0, "ocr_crops_per_vehicle": float(d)}


def _init():
    _G["train"] = TableV3.load(TABLE, "train", probe_paths=PROBES)
    _G["test"] = TableV3.load(TABLE, "test", probe_paths=PROBES)
    _G["profile"] = JCAB(_G["train"]).profile


def cfg_for(split, d, dyn):
    return dataclasses.replace(config("probe", split, False, scenarios=DYN[dyn], deadline_ms=DEADLINE), **ocr_kwargs(d))


def episode(table, cfg, n, seed, pol):
    sim = SimV3(table, cfg, seed)
    sim.reset(n)
    secs = []
    for _ in range(cfg.episode_decisions):
        secs += sim.step(pol(sim))
    m = metrics(secs, f1_target=.70)
    return {k: m[k] for k in KEYS} | {k: float(np.mean([sec[0][f] for sec in secs])) for k, f in (
        ("down", "downstream"), ("gpu", "gpu"), ("ds_served", "ds_served"), ("ds_slow", "ds_slow"))}


def future_caps(sim):
    """Capacity per model over the next 5 s: the oracle knows every camera's new vehicles ahead."""
    d = sim.cfg.ocr_ms_per_crop * sim.cfg.ocr_crops_per_vehicle
    ahead = [s.lookahead(sim.cfg.decision_s) for s in sim.streams]
    veh = np.mean([sum(a[min(t, len(a) - 1)].get("new", 0) for a in ahead) for t in range(sim.cfg.decision_s)])
    if sim.cfg.ocr_read_ms:  # live model: OCR stretches latency; throughput only by slow^exponent (as drl/sim_v3)
        busy = ocr_busy(sim.ocr_crops * veh, sim.ocr_workers, sim.cfg)[0]
        slow = (1 / (1 - min(.9, sim.cfg.ocr_gpu_hold * busy))) ** sim.cfg.ocr_cap_exponent
    else:
        slow = 1 + sim.cfg.det_stretch_per_ocr_share * d / 1000 * veh  # same stretch as drl/sim_v3
    return {m: c * sim.gpu_eff * sim.bg_base * sim.thermal_factor() / slow for m, c in sim.cfg.capacity.items()}


def mpc(sim):
    from drl.allocator import allocate, profile_scores
    from drl.train_v3 import ACTIONS
    return allocate(sim, profile_scores(sim, _G["profile"], ACTIONS), ACTIONS)


def greedy(cap):
    def pol(sim):
        c = future_caps(sim)
        gains = []
        for i, s in enumerate(sim.streams):
            ahead = s.lookahead(sim.cfg.decision_s)
            base = np.mean([x[(Y11, 15)] for x in ahead])
            gains += [(np.mean([x[(m, f)] for x in ahead]) - base, i, m, f) for m in (DX, RT) for f in (15, 10)]
        acts, used = [(Y11, 15)] * len(sim.streams), sim._probe_load(c)
        for gain, i, m, f in sorted(gains, reverse=True):
            if gain > 0 and acts[i][0] not in CLOUD and used + f / c[m] <= cap:
                acts[i], used = (m, f), used + f / c[m]
        return acts
    return pol


def rescue(cap, model):
    def pol(sim):
        c = future_caps(sim)
        edge = {i: np.mean([x[(Y11, 15)] for x in s.lookahead(sim.cfg.decision_s)]) for i, s in enumerate(sim.streams)}
        acts, used = [(Y11, 15)] * len(sim.streams), sim._probe_load(c)
        for i in sorted(edge, key=edge.get):
            if edge[i] >= .6:
                break
            fps = next((f for f in (15, 10, 5) if used + f / c[model] <= cap), None)
            if fps:
                acts[i], used = (model, fps), used + fps / c[model]
        return acts
    return pol


def tune(d, dyn):
    """Static rule and theta tuned on train for this cell (mean TUNE_KEY over N 6/10/14, 3 episodes)."""
    cfg = cfg_for("train", d, dyn)
    score = lambda pol: np.mean([episode(_G["train"], cfg, n, s, pol)[TUNE_KEY] for n in (6, 10, 14) for s in range(3)])  # noqa: E731
    static = max(((x / 10, m, f) for x in range(0, 11, 2) for m in (DX, RT) for f in (15, 10)), key=lambda g: score(rules(*g)))
    theta, cap = max(itertools.product(THETAS, CAPS), key=lambda tc: score(HeuristicThreshold(_G["profile"], *tc)))
    return static, (theta, cap)


def task(args):
    d, dyn, static, theta, n, seed = args
    cfg = cfg_for("test", d, dyn)
    test = _G["test"]
    base = {"static": episode(test, cfg, n, seed, rules(*static)),
            "thresh": episode(test, cfg, n, seed, HeuristicThreshold(_G["profile"], *theta)),
            "net": episode(test, cfg, n, seed, heuristic_net)}
    if LIVE:  # untuned "every stream on RT-DETR@15": beat the train-tuned static rule on several cells (30/10)
        base["allrt"] = episode(test, cfg, n, seed, rules(1.0, RT, 15))
        base["mpc"] = episode(test, cfg, n, seed, mpc)  # round 21: model-based scores + the central allocator
    orc = [episode(test, cfg, n, seed, p) for p in [greedy(c) for c in CAPS] + [rescue(c, m) for c in CAPS for m in (DX, RT)]]
    best = {}
    for k in KEYS:  # never worse than a baseline it could copy; OCR must keep up (served >= 99%) when any plan does
        cand = orc + list(base.values())
        ok = [o for o in cand if o["ds_served"] >= .99] or cand
        best[k] = max(ok, key=lambda o: o[k])
    drl = []
    for path in ACTORS:  # same cell and episode; the actor sees its own state variant
        from drl.train_v3 import load_policy
        pol, variant = _G.setdefault(path, load_policy(path))
        dcfg = dataclasses.replace(config(variant, "test", False, scenarios=DYN[dyn], deadline_ms=DEADLINE), **ocr_kwargs(d))
        drl.append(episode(test, dcfg, n, seed, pol))
    return d, dyn, n, seed, base, best, drl


def drl_cells(r, key, best, oracle, scale):
    """DRL mean +- std over seeds (each seed = mean of the episodes), vs the best baseline, share of the oracle gap."""
    if not r[0][6]:
        return "| - | - | - | - "
    seeds = np.mean([[e[key] for e in x[6]] for x in r], axis=0)
    m = float(np.mean(seeds))
    gap = (m - best) / (oracle - best) if oracle > best else float("nan")
    served = np.mean([e["ds_served"] for x in r for e in x[6]])
    return (f"| {m * scale:.1f} +- {np.std(seeds, ddof=1) * scale:.1f} | {(m / best - 1) * 100:+.1f}% "
            f"| {gap * 100:.0f}% | {served:.2f} ")


def main():
    with mp.Pool(14, initializer=_init) as pool:
        cells = list(itertools.product(DOWN, DYN))
        tuned = dict(zip(cells, pool.starmap(tune, cells)))
        for c, (st, th) in tuned.items():
            print(f"tuned on train {c}: static {st[0]:.0%} {st[1][:2]}@{st[2]}, theta {th[0]} GPU cap {th[1]}", flush=True)
        jobs = [(d, dyn, *tuned[(d, dyn)], n, s) for d, dyn in cells for n in NS for s in range(EPISODES)]
        rows = pool.map(task, jobs, chunksize=1)
    names = {"ta200": f"TA@{DEADLINE:.0f} x100", "fps_goodput_c60": "constrained FPS goodput (F1 >= .60)",
             "fps_goodput_c70": "constrained FPS goodput (F1 >= .70)", "fps_goodput_c75": "constrained FPS goodput (F1 >= .75)"}
    for key, name in ((k, names[k]) for k in KEYS):
        print(f"\n### {name}: oracle headroom over the best baseline (per cell, mean of {EPISODES} test episodes)")
        print(f"| OCR {'(reads/vehicle, workers, VN share)' if LIVE else 'crops/vehicle'} | dynamics | N | downstream GPU share | static | thresh | net{' | all RT-DETR | MPC' if LIVE else ''} | oracle | headroom "
              "| downstream slowdown: best baseline / oracle | downstream served | DRL (seeds) | DRL vs best | gap closed "
              "| DRL OCR served |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for d, dyn in itertools.product(DOWN, DYN):
            for n in NS:
                r = [x for x in rows if x[:3] == (d, dyn, n)]
                b = {p: np.mean([x[4][p][key] for x in r]) for p in r[0][4]}
                o = np.mean([x[5][key][key] for x in r])
                bp = max(b, key=b.get)
                scale = 100 if key == "ta200" else 1
                ag = lambda f, src: np.mean([src(x)[f] for x in r])  # noqa: E731
                print(f"| {d} | {dyn} | {n} | {ag('down', lambda x: x[4]['thresh']):.2f} | "
                      + " | ".join(f"{b[p] * scale:.1f}" for p in b)
                      + f" | {o * scale:.1f} | **{(o / b[bp] - 1) * 100:+.1f}%** "
                      f"| {ag('ds_slow', lambda x: x[4][bp]):.2f}x / {ag('ds_slow', lambda x: x[5][key]):.2f}x "
                      f"| {ag('ds_served', lambda x: x[4][bp]):.2f} / {ag('ds_served', lambda x: x[5][key]):.2f} "
                      + drl_cells(r, key, b[bp], o, scale) + "|")


if __name__ == "__main__":
    main()
