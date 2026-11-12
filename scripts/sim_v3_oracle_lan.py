"""Headroom on LAN (round 15b): how much any policy can beat the tuned threshold heuristic when every Edge is on LAN.

  PYTHONPATH=. python -B scripts/sim_v3_oracle_lan.py

Oracle per test episode, best in hindsight of:
  - greedy content oracle (knows each stream's next 5 s F1; largest D-FINE-X / RT-DETR gains first) under a GPU cap,
  - rescue oracle (knows each stream's whole-episode Edge F1): streams below F1 .6 on Edge go to the Cloud model, worst
    first, at the highest FPS the GPU cap allows; the rest YOLO11s@15,
  GPU caps .8 / .9 / 1.0. Reported with the threshold / net heuristics and both DRL variants (5 seeds each).
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import CLOUD, SimV3, TableV3, metrics  # noqa: E402
from drl.train_v3 import config  # noqa: E402
from scripts.sim_v3_baselines import TABLE, PROBES, HeuristicThreshold, JCAB, heuristic_net, Y11, DX, RT  # noqa: E402

THETA = .05  # threshold heuristic, tuned on train for LAN (eval15b_lan.log)
NS, EPISODES = (6, 12, 20, 30), 10
ACTORS = {v: [f"results/drl_v3_cal/actor_{v}_s{s}.pt" for s in range(5)] for v in ("probe_net_5g_fps3", "probe_lan_fps3")}
_G = {}


def _init():
    _G["train"] = TableV3.load(TABLE, "train", probe_paths=PROBES)
    _G["test"] = TableV3.load(TABLE, "test", probe_paths=PROBES)
    _G["ht"] = HeuristicThreshold(JCAB(_G["train"]).profile, THETA)


def caps(sim):
    return {m: c * sim.gpu_eff * sim.background for m, c in sim.cfg.capacity.items()}


def greedy(cap):
    def pol(sim):
        c = caps(sim)
        gains = []
        for i, s in enumerate(sim.streams):
            ahead = s.lookahead(sim.cfg.decision_s)
            base = np.mean([x[(Y11, 15)] for x in ahead])
            gains += [(np.mean([x[(m, 15)] for x in ahead]) - base, i, m) for m in (DX, RT)]
        acts, used = [(Y11, 15)] * len(sim.streams), sim._probe_load(c)
        for gain, i, m in sorted(gains, reverse=True):
            if gain > 0 and acts[i][0] not in CLOUD and used + 15 / c[m] <= cap:
                acts[i], used = (m, 15), used + 15 / c[m]
        return acts
    return pol


def rescue(cap, model):
    def pol(sim):
        c = caps(sim)
        edge = {i: np.mean([x[(Y11, 15)] for x in s.lookahead(10 ** 4)]) for i, s in enumerate(sim.streams)}
        acts, used = [(Y11, 15)] * len(sim.streams), sim._probe_load(c)
        for i in sorted(edge, key=edge.get):
            if edge[i] >= .6:
                break
            fps = next((f for f in (15, 10, 5) if used + f / c[model] <= cap), None)
            if fps:
                acts[i], used = (model, fps), used + fps / c[model]
        return acts
    return pol


def episode(cfg, n, seed, pol):
    sim = SimV3(_G["test"], cfg, seed)
    sim.reset(n)
    secs = []
    for _ in range(cfg.episode_decisions):
        secs += sim.step(pol(sim))
    return metrics(secs, f1_target=.70)["fps_goodput_c60"]


def task(args):
    n, seed = args
    cfg = config("probe", "test", False)
    cands = [greedy(c) for c in (.8, .9, 1.0)] + [rescue(c, m) for c in (.8, .9, 1.0) for m in (DX, RT)]
    scores = [episode(cfg, n, seed, p) for p in cands]
    from drl.train_v3 import load_policy
    drl = {}
    for v, paths in ACTORS.items():
        vals = []
        for p in paths:
            pol, variant = _G.setdefault(p, load_policy(p))
            vals.append(episode(config(variant, "test", False), n, seed, pol))
        drl[v] = float(np.mean(vals))
    return (n, seed, max(scores), int(np.argmax(scores)), episode(cfg, n, seed, _G["ht"]),
            episode(cfg, n, seed, heuristic_net), drl["probe_net_5g_fps3"], drl["probe_lan_fps3"])


def main():
    with mp.Pool(14, initializer=_init) as pool:
        rows = pool.map(task, [(n, s) for n in NS for s in range(EPISODES)], chunksize=1)
    names = ["greedy .8", "greedy .9", "greedy 1.0"] + [f"rescue {c} {m[:2]}" for c in (.8, .9, 1.0) for m in (DX, RT)]
    print("| N | Max (N x 15) | Oracle | Threshold heur. | Oracle headroom | Net heur. | DRL LAN+5G train | DRL LAN-only |")
    print("|---|---|---|---|---|---|---|---|")
    for n in NS:
        r = [x for x in rows if x[0] == n]
        o, ht, hn, d5, dl = (np.mean([x[k] for x in r]) for k in (2, 4, 5, 6, 7))
        print(f"| {n} | {15 * n} | {o:.1f} | {ht:.1f} | {(o / ht - 1) * 100:+.1f}% | {hn:.1f} | {d5:.1f} | {dl:.1f} |")
    print("oracle picks:", {names[k]: sum(x[3] == k for x in rows) for k in range(len(names))})


if __name__ == "__main__":
    main()
