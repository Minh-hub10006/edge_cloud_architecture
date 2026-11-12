"""Headroom of the 5G part (round 15b): how much any 5G-site policy can beat "5G sites -> YOLO11s Edge".

  PYTHONPATH=. python -B scripts/sim_v3_oracle5g.py

LAN sites always run the tuned threshold heuristic, so only the 5G sites differ. Oracle, per test episode and 5G
site (best in hindsight, knows the future uplink and every stream's true F1): the best of
  - 64 static plans (each of the site's 3 streams: Edge YOLO11s@15 or D-FINE-X @5/10/15 for the whole episode),
  - dynamic plans: every 5 s, streams by true F1 gain over the next 5 s, D-FINE-X at the highest FPS that fits the
    minimum uplink of the next 5 s / margin (margins 1, 1.2, 1.5, 2, 3), all streams or only Edge-F1 < .6 ones;
sites optimised one at a time (others on Edge), then combined; never below the Edge-only plan.
Also reported: Edge-only, threshold heuristic, DRL (5 seeds). 5G share of Edge sites: 50% (training) and 25%.
"""
import dataclasses
import itertools
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import SimV3, TableV3, metrics  # noqa: E402
from drl.train_v3 import config  # noqa: E402
from scripts.sim_v3_baselines import TABLE, PROBES, HeuristicThreshold, JCAB, Y11, DX  # noqa: E402

THETA = .12  # threshold heuristic, tuned on train for LAN+5G (eval15b_lan5g.log)
BITS = 76 * 8 * 1024 / 1e6  # Mbit per JPEG (SimV3Config.jpeg_kb)
NS, EPISODES, MIXES = (6, 12, 20, 30), 10, (.5, .25)
ACTORS = [f"results/drl_v3_cal/actor_probe_net_5g_fps3_s{s}.pt" for s in range(5)]
_G = {}


def _init():
    _G["train"] = TableV3.load(TABLE, "train", probe_paths=PROBES)
    _G["test"] = TableV3.load(TABLE, "test", probe_paths=PROBES)
    _G["ht"] = HeuristicThreshold(JCAB(_G["train"]).profile, THETA)


def cfg_for(mix):
    return dataclasses.replace(config("probe", "test", True, traces="5g"), net_mix={"lan": 1 - mix, "5g": mix})


def episode(cfg, n, seed, site_plan):
    """site_plan(sim, host, streams) -> actions for those streams, or None = Edge. LAN sites: threshold heuristic."""
    sim = SimV3(_G["test"], cfg, seed)
    sim.reset(n)
    secs = []
    for _ in range(cfg.episode_decisions):
        acts = list(_G["ht"](sim))
        for h, (kind, _, _) in sim.links.items():
            if kind == "lan":
                continue
            mine = [i for i in range(n) if sim.host[i] == h]
            plan = site_plan(sim, h, mine) if site_plan else None
            for i in mine:
                acts[i] = plan[i] if plan and i in plan else (Y11, 15)
        secs += sim.step(acts)
    return metrics(secs, f1_target=.70)["fps_goodput_c60"]


def future_ul(sim, h):
    kind, k, cursor = sim.links[h]
    ul = sim.cfg.net_traces[kind][k][0]
    return min(float(ul[(cursor + j) % len(ul)]) for j in range(sim.cfg.decision_s)) * sim.ul_scale[h]


def dynamic(margin, only_hard, hosts):
    def plan(sim, h, mine):
        if h not in hosts:
            return None
        budget, out = future_ul(sim, h) / margin, {}
        gain = {}
        for i in mine:
            ahead = sim.streams[i].lookahead(sim.cfg.decision_s)
            gain[i] = np.mean([c[(DX, 15)] - c[(Y11, 15)] for c in ahead])
            if only_hard and np.mean([c[(Y11, 15)] for c in sim.streams[i].lookahead(10 ** 4)]) >= .6:
                gain[i] = -1
        for i in sorted(mine, key=lambda i: -gain[i]):
            if gain[i] <= 0:
                continue
            fps = next((f for f in (15, 10, 5) if f * BITS <= budget), None)
            if fps:
                out[i], budget = (DX, fps), budget - fps * BITS
        return out
    return plan


def static(choice, hosts):
    def plan(sim, h, mine):
        return {i: c for i, c in zip(mine, choice) if c} if h in hosts else None
    return plan


def task(args):
    mix, n, seed = args
    cfg = cfg_for(mix)
    sim = SimV3(_G["test"], cfg, seed)
    sim.reset(n)
    sites = [h for h, (k, _, _) in sim.links.items() if k != "lan"]
    edge = episode(cfg, n, seed, None)
    best = {}
    for h in sites:
        size = sum(1 for x in sim.host if x == h)
        cands = [static(c, {h}) for c in itertools.product((None, (DX, 5), (DX, 10), (DX, 15)), repeat=size)]
        cands += [dynamic(m, o, {h}) for m in (1.0, 1.2, 1.5, 2.0, 3.0) for o in (False, True)]
        scores = [episode(cfg, n, seed, c) for c in cands]
        best[h] = cands[int(np.argmax(scores))] if max(scores) > edge else None

    def combined(sim_, h, mine):
        return best[h](sim_, h, mine) if best.get(h) else None
    oracle = max(edge, episode(cfg, n, seed, combined)) if sites else edge
    heur = episode_policy(cfg, n, seed, _G["ht"])
    drl = [episode_policy(cfg, n, seed, actor(p), variant=True) for p in ACTORS]
    return mix, n, seed, len(sites), edge, oracle, heur, float(np.mean(drl))


def actor(path):
    from drl.train_v3 import load_policy
    if path not in _G:
        _G[path] = load_policy(path)
    return _G[path]


def episode_policy(cfg, n, seed, pol, variant=False):
    if variant:  # DRL: the simulator replays its state variant (probes); same links and seeds
        pol, v = pol
        cfg = dataclasses.replace(config(v, "test", True, traces="5g"), net_mix=cfg.net_mix)
    sim = SimV3(_G["test"], cfg, seed)
    sim.reset(n)
    secs = []
    for _ in range(cfg.episode_decisions):
        secs += sim.step(pol(sim))
    return metrics(secs, f1_target=.70)["fps_goodput_c60"]


def main():
    jobs = [(m, n, s) for m in MIXES for n in NS for s in range(EPISODES)]
    with mp.Pool(14, initializer=_init) as pool:
        rows = pool.map(task, jobs, chunksize=1)
    print("| 5G share | N | 5G sites (mean) | Edge-only | Oracle | Oracle headroom | Threshold heur. | DRL (5 seeds) |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MIXES:
        for n in NS:
            r = np.array([x[3:] for x in rows if x[0] == m and x[1] == n])
            sites, edge, orc, heur, drl = r.mean(0)
            print(f"| {m:.0%} | {n} | {sites:.1f} | {edge:.1f} | {orc:.1f} | {(orc / edge - 1) * 100:+.1f}% | "
                  f"{heur:.1f} | {drl:.1f} |")


if __name__ == "__main__":
    main()
