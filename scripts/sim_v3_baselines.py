"""Baselines in drl/sim_v3 at growing N, every headline candidate metric (docs/PLAN_DRL_V3B_VI.md P3, metric choice).

  python -B scripts/sim_v3_baselines.py [--split test] [--n 9 15 21 30] [--seeds 10] [--actors A.pt ...]

Baselines: always <model>@15; rules(k, cloud model, FPS): the k streams on the strongest Edges (247, 246, 252
order) on the Cloud model, the rest YOLO11s@15, k and the Cloud model picked per N on the train split by goodput;
oracle: knows each stream's next 5 s F1 and greedily gives the Cloud models to the largest gains while the GPU stays
below 90%. heuristic_load (v1 "rules" style): one more stream (strongest Edge first) to D-FINE-X while the GPU is
below 70%, one back to YOLO11s above 90%. heuristic_probe: streams ranked by their last probe, D-FINE-X down the
ranking while the predicted GPU stays below 90%, then RT-DETR, the rest YOLO11s (same information as the probe
DRL, no learning). --actors adds trained DRL actors (drl/train_v3.py), greedy; each runs with its own state variant (the
probe variant's simulator replays probes, which also load the GPU).
"""
import argparse
import itertools
import math
import random
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drl.sim_v3 import CLOUD, SimV3, SimV3Config, TableV3, metrics  # noqa: E402
from drl.train_v3 import PROBES, config, load_policy  # noqa: E402

TABLE = "results/table_fps_all.jsonl"
THETAS = (0.0, .02, .05, .08, .12, .16, .22)
EDGE_RANK = {"edge-247": 0, "edge-246": 1, "edge-252": 2}
Y11, DX, RT = "yolo11s_edge", "dfine_x_cloud", "rtdetr_cloud"


def fixed(mode, fps=15):
    return lambda sim: [(mode, fps)] * len(sim.streams)


def rules(k, cloud_mode, fps):
    """k streams (int) or a share of the streams (float < 1, Chameleon-style one static configuration)."""
    def act(sim):
        order = sorted(range(len(sim.streams)), key=lambda i: (EDGE_RANK[sim.streams[i].edge.name], i))
        top = set(order[:k if isinstance(k, int) else round(k * len(sim.streams))])
        return [(cloud_mode, fps) if i in top else (Y11, 15) for i in range(len(sim.streams))]
    return act


class HeuristicLoad:
    def __init__(self):
        self.sim, self.k = None, 0

    def __call__(self, sim):
        if sim is not self.sim:  # new episode
            self.sim, self.k = sim, 0
        elif sim.gpu_util < .7:
            self.k = min(len(sim.streams), self.k + 1)
        elif sim.gpu_util > .9:
            self.k = max(0, self.k - 1)
        return rules(self.k, DX, 15)(sim)


def heuristic_probe(sim):
    cfg = sim.cfg
    caps = {m: c * cfg.gpu_efficiency * sim.background for m, c in cfg.capacity.items()}
    order = sorted(range(len(sim.streams)), key=lambda i: -(sim.streams[i].probe_last or 0.0))
    acts, used = [(Y11, 15)] * len(sim.streams), sim._probe_load(caps)
    for mode in (DX, RT):
        for i in order:
            if acts[i][0] == Y11 and used + 15 / caps[mode] <= .9:
                acts[i], used = (mode, 15), used + 15 / caps[mode]
    return acts


def heuristic_net(sim, gpu_cap=.9):
    """Network-aware probe heuristic: the probe heuristic plus each site's last measured uplink and transport delay
    (the information the *_net DRL state carries) and the FPS knob. Hardest streams first; a stream gets the first
    (model, FPS) in DX/RT x 15/10/5 that keeps the predicted GPU <= .9, its site's uplink (x1.2 margin) and the
    predicted Cloud latency (join + transport + transfer + GPU) <= 150 ms; otherwise YOLO11s@15."""
    cfg = sim.cfg
    caps = {m: c * cfg.gpu_efficiency * sim.background for m, c in cfg.capacity.items()}
    bits = cfg.jpeg_kb * 8 * 1024 / 1e6
    budget = {}
    for i, s in enumerate(sim.streams):
        last = s.last or {}
        if last.get("mobile"):
            budget.setdefault(sim.host[i], last.get("uplink", 0.0) / 1.2 - sim._probe_rate() * bits *
                              sum(1 for j, h in enumerate(sim.host) if h == sim.host[i]))
    order = sorted(range(len(sim.streams)), key=lambda i: -(sim.streams[i].probe_last or 0.0))
    acts, used = [(Y11, 15)] * len(sim.streams), sim._probe_load(caps)
    for i in order:
        s, last = sim.streams[i], sim.streams[i].last or {}
        for mode, fps in itertools.product((DX, RT), (15, 10, 5)):
            need = fps * bits
            ul = budget.get(sim.host[i])
            tx = 0.0 if ul is None else 1000 * bits / max(last.get("uplink", 1e-3), 1e-3)
            latency = s.edge.join_ms + last.get("net_rtt", 0.0) + tx + 1000 / caps[mode]
            if used + fps / caps[mode] <= gpu_cap and (ul is None or need <= ul) and latency <= 150:
                acts[i], used = (mode, fps), used + fps / caps[mode]
                if ul is not None:
                    budget[sim.host[i]] = ul - need
                break
    return acts


class JCAB:
    """JCAB (Wang, Zhang et al., INFOCOM 2020 / IEEE/ACM ToN 2022) adapted: per decision, choose every stream's
    configuration (model, FPS) by Lyapunov drift-plus-penalty, solved by Markov approximation (Gibbs-style random
    single-stream moves). Same information as the DRL state: probe, site uplink / transport delay, GPU capacity.
      utility(i, a) = V x acc_i(a) - Q_i x late_i(a);   s.t. GPU <= .95, each mobile site's uplink.
    acc_i(a): offline profile on the train split, per (model, FPS) linear in the probe (F1 ~ c0 + c1 x probe).
    late_i(a): 1 if the predicted Cloud latency (join + transport + transfer + GPU service) > deadline - 20 ms, else 0.
    Q_i <- max(0, Q_i + late share of the last 5 s - .05)  (virtual queue: long-term late share <= 5 %)."""

    def __init__(self, train_table, V=10.0, steps=300, beta=50.0, seed=0):
        self.V, self.steps, self.beta, self.rng = V, steps, beta, random.Random(seed)
        self.actions = [(m, f) for m in ("yolo_edge", Y11, RT, DX) for f in (15, 10, 5)]
        self.profile = {}
        for a in self.actions:
            xs, ys = [], []
            for cells in train_table.seq.values():
                for c in cells:
                    if c.get("probe") is not None:
                        xs.append(float(np.mean(c["probe"]))); ys.append(c[a])
            x, y = np.array(xs), np.array(ys)  # least squares by hand: polyfit's LAPACK clashes with torch's OpenMP
            c1 = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
            self.profile[a] = (float(y.mean() - c1 * x.mean()), c1)
        self.sim, self.Q = None, None

    def __call__(self, sim):
        cfg = sim.cfg
        if sim is not self.sim:
            self.sim, self.Q = sim, [0.0] * len(sim.streams)
        else:
            for i, s in enumerate(sim.streams):
                last = s.last or {"done": 1.0, "on_time": 1.0}
                self.Q[i] = max(0.0, self.Q[i] + (1 - last["done"] * last["on_time"]) - .05)
        caps = {m: c * cfg.gpu_efficiency * sim.background for m, c in cfg.capacity.items()}
        bits = cfg.jpeg_kb * 8 * 1024 / 1e6
        uplink = {}
        for i, s in enumerate(sim.streams):
            if (s.last or {}).get("mobile"):
                uplink.setdefault(sim.host[i], s.last.get("uplink", 0.0))

        def utility(i, a):
            s, last = sim.streams[i], sim.streams[i].last or {}
            c0, c1 = self.profile[a]
            acc = c0 + c1 * (s.probe_last if s.probe_last is not None else 0.0)
            late = 0.0
            if a[0] in CLOUD:
                tx = 1000 * bits / max(last.get("uplink", 1e9), 1e-3) if sim.host[i] in uplink else 0.0
                late = float(s.edge.join_ms + last.get("net_rtt", 0.0) + tx + 1000 / caps[a[0]] > cfg.deadline_ms - 20)
            return self.V * acc - self.Q[i] * late

        def feasible(acts):
            if sum(f / caps[m] for m, f in acts if m in CLOUD) + sim._probe_load(caps) > .95:
                return False
            for h, ul in uplink.items():
                need = sum(f * bits for i, (m, f) in enumerate(acts) if sim.host[i] == h and m in CLOUD)
                if need > ul:
                    return False
            return True

        acts = [(Y11, 15)] * len(sim.streams)  # always feasible start
        score = [utility(i, a) for i, a in enumerate(acts)]
        for _ in range(self.steps):
            i = self.rng.randrange(len(acts))
            a = self.rng.choice(self.actions)
            trial = acts[:i] + [a] + acts[i + 1:]
            if not feasible(trial):
                continue
            delta = utility(i, a) - score[i]
            if self.rng.random() < 1 / (1 + math.exp(-max(-50.0, min(50.0, self.beta * delta)))):
                acts, score[i] = trial, utility(i, a)
        return acts


class HeuristicThreshold:
    """Network-aware heuristic that only offloads when it pays: a stream goes to a Cloud model only if its predicted
    F1 gain over YOLO11s@15 (offline profile on train, linear in the probe, as JCAB) is >= theta; then the
    heuristic_net feasibility checks (GPU <= .9, site uplink, latency <= 150 ms). theta tuned on train; a theta sweep
    gives its accuracy-resource Pareto curve."""

    def __init__(self, profile, theta, gpu_cap=.9):  # gpu_cap tuned too where the Cloud is shared (OCR)
        self.profile, self.theta, self.gpu_cap = profile, theta, gpu_cap

    def __call__(self, sim):
        acts = heuristic_net(sim, self.gpu_cap)
        for i, s in enumerate(sim.streams):
            mode, fps = acts[i]
            if mode in CLOUD:
                x = s.probe_last if s.probe_last is not None else 0.0
                c0, c1 = self.profile[(mode, fps)]
                b0, b1 = self.profile[(Y11, 15)]
                if (c0 + c1 * x) - (b0 + b1 * x) < self.theta:
                    acts[i] = (Y11, 15)
        return acts


class MobileEdge(HeuristicThreshold):
    """Mobile sites never offload (YOLO11s@15 on the Edge), LAN sites run the threshold heuristic. Round 15b: every
    probe_net_5g_fps3 seed learned exactly this rule and this baseline tied it."""

    def __call__(self, sim):
        return [(Y11, 15) if sim.links[sim.host[i]][0] != "lan" else a for i, a in enumerate(super().__call__(sim))]


def oracle(sim):
    cfg = sim.cfg
    caps = {m: c * cfg.gpu_efficiency * sim.background for m, c in cfg.capacity.items()}
    gains = []
    for i, s in enumerate(sim.streams):
        ahead = s.lookahead(cfg.decision_s)
        base = np.mean([c[(Y11, 15)] for c in ahead])
        for mode in (DX, RT):
            gains.append((np.mean([c[(mode, 15)] for c in ahead]) - base, i, mode))
    acts, used = [(Y11, 15)] * len(sim.streams), 0.0
    for gain, i, mode in sorted(gains, reverse=True):
        if gain <= 0 or acts[i][0] in CLOUD or used + 15 / caps[mode] > .9:
            continue
        acts[i], used = (mode, 15), used + 15 / caps[mode]
    return acts


def run(table, policy, n, seeds, cfg):
    out = []
    for seed in range(seeds):
        sim = SimV3(table, cfg, seed)
        sim.reset(n)
        secs = []
        for _ in range(cfg.episode_decisions):
            secs += sim.step(policy(sim))
        m = metrics(secs, f1_target=.70)
        vn = [r for s in secs for r in s if r["dataset"] == "vn_hard"]
        m["f1_vn"] = float(np.mean([r["f1"] for r in vn])) if vn else float("nan")
        m["ta_vn"] = float(np.mean([r["ta"] for r in vn])) if vn else float("nan")
        recs = [r for s in secs for r in s]  # policy behaviour: share of stream-seconds per model, mean FPS
        m.update({f"mix_{x}": float(np.mean([r["mode"] == x for r in recs])) for x in (DX, RT, Y11, "yolo_edge")})
        m["fps_mean"] = float(np.mean([r["fps"] for r in recs]))
        m["lat_p95"] = float(np.percentile([r["latency"] for r in recs], 95))  # median path latency per second
        # mobile data sent: Cloud frames of streams on 5G/4G sites (frames served x JPEG size), Mbit/s summed over sites
        bits = cfg.jpeg_kb * 8 * 1024 / 1e6
        m["mobile_mbps"] = float(np.mean([sum(r["fps"] * r["done"] * bits for r in s if r["uplink"] is not None
                                              and r["mode"] in CLOUD) for s in secs]))
        out.append(m)
    agg = {k: float(np.mean([m[k] for m in out])) for k in out[0]}
    agg["episodes"] = [{k: m[k] for k in ("fps_goodput_c60", "drop", "f1", "gpu_util", "ta200", "on_time")}
                       for m in out]  # error bars
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, nargs="+", default=[9, 15, 21, 30])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--actors", nargs="*", default=[])
    p.add_argument("--no-baselines", action="store_true", help="only the actors")
    p.add_argument("--net", action="store_true", help="Edge sites on LAN / 4G / 5G (drl/net_traces.py), every policy")
    p.add_argument("--ul5g", type=float, default=None, help="sensitivity: 5G uplink median Mbit/s (default 50)")
    p.add_argument("--gpu-eff", type=float, default=None, help="sensitivity: gpu_efficiency (default .8)")
    p.add_argument("--policies", nargs="*", default=None, help="only these baseline names (prefix match)")
    p.add_argument("--traces", choices=("ucc", "mahimahi", "5g"), default=None, help="mobile uplink traces with --net")
    p.add_argument("--rtt", type=float, default=None, help="mahimahi extra RTT ms (default 50)")
    p.add_argument("--scenario", nargs="*", default=None, help="LAN dynamics: surge diurnal thermal edgeslow")
    p.add_argument("--episode-decisions", type=int, default=None)
    p.add_argument("--deadline", type=float, default=None, help="deadline ms for TA / on-time (default 200)")
    p.add_argument("--pareto", action="store_true", help="also sweep the threshold heuristic's theta (Pareto curve)")
    p.add_argument("--objective", default="ta200", help="what baselines are tuned for on train: ta200 | fps_goodput_c60")
    a = p.parse_args()
    over = {k: v for k, v in (("ul5g", a.ul5g), ("gpu_eff", a.gpu_eff), ("traces", a.traces), ("rtt_ms", a.rtt),
                              ("scenarios", a.scenario), ("episode_decisions", a.episode_decisions),
                              ("deadline_ms", a.deadline)) if v is not None}
    cfg = config("none", "train", a.net, **over)  # rules tuned on train links
    cfg_eval = config("none", a.split, a.net, **over)
    train = TableV3.load(TABLE, "train", probe_paths=PROBES)  # probes: JCAB profile fit
    test = TableV3.load(TABLE, a.split, probe_paths=PROBES)
    rows = []
    # One static configuration for every N (Chameleon's "best one-time configuration"): share of streams on the
    # Cloud model at FPS, tuned once on train over N = 6..30 by mean TA@200.
    if not a.no_baselines:
        grid = [(x / 10, m, f) for x, m, f in itertools.product(range(1, 10), (DX, RT), (15, 10))]
        static = max(grid, key=lambda g: np.mean([run(train, rules(*g), n, 3, cfg)[a.objective] for n in (6, 12, 18, 24, 30)]))
        print(f"static rule (tuned once): {static[0]:.0%} of streams on {static[1]}@{static[2]}", flush=True)
        # JCAB's accuracy/latency trade-off V tuned once on train (N 9-21, train links), like the static rule
        cfg_p = config("probe", "train", a.net, **over)
        jcab_v = max((.3, 1.0, 3.0, 10.0, 30.0), key=lambda v: np.mean(
            [run(train, JCAB(train, V=v), n, 2, cfg_p)[a.objective] for n in (9, 15, 21)]))
        print(f"JCAB V tuned on train: {jcab_v}", flush=True)
        jcab_profile = JCAB(train).profile
        th_best = max(THETAS, key=lambda th: np.mean(
            [run(train, HeuristicThreshold(jcab_profile, th), n, 2, cfg_p)[a.objective] for n in (9, 15, 21)]))
        print(f"threshold heuristic theta tuned on train ({a.objective}): {th_best}", flush=True)
    for n in a.n:
        # rules tuned on train by goodput: k streams on the Cloud model at fps
        grid = [(k, m, f) for k, m, f in itertools.product(range(0, n + 1, max(1, n // 10)), (DX, RT), (15, 10))]
        best = (0, DX, 15) if a.no_baselines else max(grid, key=lambda g: run(train, rules(*g), n, 4, cfg)["goodput"])
        policies = {"always_Y8": fixed("yolo_edge"), "always_Y11": fixed(Y11), "always_RT": fixed(RT),
                    "always_DX": fixed(DX), f"rules_knowsN k={best[0]} {best[1][:2]}@{best[2]}": rules(*best),
                    "oracle_content": oracle}
        runs = [] if a.no_baselines else [(name, pol, cfg_eval) for name, pol in policies.items()]
        if not a.no_baselines:
            runs.insert(4, (f"rules_static {static[0]:.0%} {static[1][:2]}@{static[2]}", rules(*static), cfg_eval))
            runs[-1:-1] = [("heuristic_load", HeuristicLoad(), cfg_eval),
                           ("heuristic_probe", heuristic_probe, config("probe", a.split, a.net, **over)),
                           ("heuristic_net", heuristic_net, config("probe", a.split, a.net, **over)),
                           (f"jcab V={jcab_v}", JCAB(train, V=jcab_v), config("probe", a.split, a.net, **over)),
                           (f"heur_thresh th={th_best}", HeuristicThreshold(jcab_profile, th_best), config("probe", a.split, a.net, **over))]
            if a.net:
                runs.append((f"mobile_edge th={th_best}", MobileEdge(jcab_profile, th_best), config("probe", a.split, a.net, **over)))
            if a.pareto:  # accuracy-resource curve of the threshold heuristic
                runs += [(f"heur_thresh_sweep th={th}", HeuristicThreshold(jcab_profile, th), config("probe", a.split, a.net, **over))
                         for th in THETAS]
        for path in a.actors:
            pol, variant = load_policy(path)
            runs.append((f"DRL {Path(path).stem}", pol,
                         config(variant, a.split, a.net or variant.endswith(("_net", "_mm")), **over)))
        if a.policies:
            runs = [r for r in runs if r[0].startswith("DRL") or any(r[0].startswith(x) for x in a.policies)]
        for name, pol, run_cfg in runs:
            m = run(test, pol, n, a.seeds, run_cfg)
            rows.append({"n": n, "policy": name, **m})
            print(f"N={n:2d} {name:28} FPSgood {m['fps_goodput']:6.1f} c70 {m['fps_goodput_c70']:6.1f} c60 {m['fps_goodput_c60']:6.1f} drop {m['drop']:.3f} ok70 {m['share_ok70']:.2f} W {m['gpu_w']:5.0f} TA/GPU {m['ta_per_gpu']:.2f} TA {m['ta200']:.3f}  goodput {m['goodput']:6.1f}  SLO {m['slo_streams']:5.1f}/{n}"
                  f"  F1 {m['f1']:.3f} on-time {m['on_time']:.3f}  VN F1 {m['f1_vn']:.3f} VN TA {m['ta_vn']:.3f}  GPU {m['gpu_util']:.2f}", flush=True)
    Path("results/drl_v3").mkdir(parents=True, exist_ok=True)
    tag = "".join(f"_{k}{v}" for k, v in over.items())
    Path(f"results/drl_v3/{'drl' if a.no_baselines else 'baselines'}_{a.split}{'_net' if a.net else ''}{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
