"""Causal bound (30/10): how much of the oracle's headroom is reachable WITHOUT seeing the future?

Each decision the planner freezes every camera at its current second (true F1 per model/FPS and new vehicles of NOW,
repeated for the next 5 s: perfect perception, no future), builds on that frozen copy exactly the plans the oracle
tries (greedy at GPU caps .6-1.0, rescue to D-FINE-X / RT-DETR at those caps) plus every baseline (train-tuned static
rule and threshold heuristic, net heuristic, all RT-DETR@15, MPC), simulates each plan 5 s on the frozen copy, and runs
the best-scoring one on the real sim. The only difference from the oracle is that it never sees the future.

Step 2 (30/10): the plan score counts each camera's whole episode so far (observed late share, the F1 the
planner believed for its past plans) plus the 5 s ahead, so the 5 % drop budget and the F1 floor hold over the episode,
not only the next 5 s; and the frozen F1 is either the truth, a probe estimate (offline profile, as JCAB) or a probe
estimate per camera type (VN / DETRAC site, known at installation).

  PYTHONPATH=. python -B scripts/sim_v3_causal_bound.py
"""
import copy
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.argv.append("--live")  # the oracle module's live grid (OCR model fitted on the testbed, c70 at 150 ms)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.sim_v3_oracle_shared as o  # noqa: E402
from drl.train_v3 import ACTIONS as _ACTIONS  # noqa: E402
o.ACTIONS_ALL = list(_ACTIONS)
from scripts.sim_v3_ocr_knob import score  # noqa: E402

CELLS = [((.45, 8, .4), "none"), ((.45, 8, .4), "all"), ((.7, 8, .4), "none"), ((.7, 8, .4), "all"),
         ((.45, 4, .7), "none"), ((.45, 8, .7), "none"), ((.45, 8, .7), "all"), ((.7, 4, .7), "none")]
NS, EPISODES = (10, 14), 5


def profiles(table):
    """F1 ~ c0 + c1 x probe per (camera type, action) and over all cameras (key None), fitted on the train split."""
    out = {}
    for ds in [None] + sorted(table.by_dataset):
        keys = table.by_dataset[ds] if ds else list(table.seq)
        cells = [c for k in keys for c in table.seq[k] if c.get("probe") is not None]
        x = np.array([float(np.mean(c["probe"])) for c in cells])
        for a in o.ACTIONS_ALL:
            y = np.array([c[a] for c in cells])
            c1 = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
            out[(ds, a)] = (float(y.mean() - c1 * x.mean()), c1)
    return out


def frozen(sim, f1="true", est=None, ocr_scale=1.0):
    """Copy of sim in which every stream stays at its current second (F1 cells and new vehicles of now).
    f1: "true" | "probe" (estimate from the probe) | "probe_site" (estimate from the probe and the camera type)."""
    f = copy.deepcopy(sim, memo={id(sim.table): sim.table})
    for i, s in enumerate(f.streams):
        now = s.cell()
        if ocr_scale != 1.0:  # robust planning: plate arrivals at a recent high, not this second's (bursty VN traffic)
            now = dict(now)
            now["new"] = now.get("new", 0) * ocr_scale
        if f1 != "true":
            x = (est or {}).get(i, s.probe_last if s.probe_last is not None else 0.0)
            ds = s.dataset if f1 == "probe_site" else None
            now = dict(now)
            for a in o.ACTIONS_ALL:
                c0, c1 = o._G["prof"][(ds, a)]
                now[a] = float(np.clip(c0 + c1 * x, 0, 1))
        s.cell = lambda now=now: now
        s.lookahead = lambda n, now=now: [now] * n
        s._jump = lambda: None  # surge/diurnal must not swap the frozen scene either
    return f


def episode_score(secs, past, f1_min=.70, episode_s=120):
    """Soft constrained FPS of the next seconds, each camera's F1 and late share projected to the END of the episode
    as if this plan were kept: so far + this plan's rate x the time left. (Fix 30/10: dividing by the time elapsed
    let one decision spend the whole 5 % late budget -- at 60 s a plan 60 % late for 5 s looked like 4.6 %.)"""
    out, h = 0.0, len(secs)
    for i in range(len(secs[0])):
        t, late_sum, f1_sum = past[i]
        left = max(h, episode_s - t)
        eff = np.mean([s[i]["fps"] * s[i]["done"] * s[i]["on_time"] for s in secs])
        late_rate = np.mean([1 - s[i]["done"] * s[i]["on_time"] for s in secs])
        f1_rate = np.mean([s[i]["f1"] for s in secs])
        late = (late_sum + late_rate * left) / (t + left)
        f1 = (f1_sum + f1_rate * left) / (t + left)
        out += eff * np.clip((f1 - (f1_min - .05)) / .10, 0, 1) * np.clip((.10 - late) / .05, 0, 1)
    return out


def update_estimates(sim, est):
    """Per camera: exponentially weighted probe (weight grows with the frames in the new probe) and its spread."""
    for i, s in enumerate(sim.streams):  # by stream index: the planner's frozen copies are other objects
        if s.probe_age == 0.0 and s.probe_last is not None:
            e, v = est.get(i, (s.probe_last, 1.0))
            a = min(.8, .35 * getattr(s, "probe_n", 1))
            new = (1 - a) * e + a * s.probe_last
            est[i] = (new, (1 - a) * v + a * (s.probe_last - e) ** 2)


def plan_probes(sim, est, budget_s):
    """Selective probing: budget_s probe-seconds this decision (= the fixed scheme's cost when it equals the Edge
    streams), one at a time to the camera whose Edge F1 estimate is nearest .70, whose probe is oldest or noisiest,
    with diminishing returns per camera; cameras on D-FINE-X need none."""
    pri = {}
    for i, s in enumerate(sim.streams):
        if s.mode == "dfine_x_cloud":
            continue
        e, v = est.get(i, (None, 1.0))
        if e is None:
            pri[i] = 3.0  # never probed
            continue
        c0, c1 = o._G["prof"][(None, ("yolo11s_edge", 15))]
        edge_f1 = c0 + c1 * e
        pri[i] = np.exp(-abs(edge_f1 - .70) / .05) + min(1.0, (s.probe_age or 0) / 30) + min(1.0, np.sqrt(v) / 5)
    plan = {}
    for _ in range(int(budget_s)):
        if not pri:
            break
        i = max(pri, key=lambda j: pri[j] / (1 + plan.get(j, 0)))
        if plan.get(i, 0) < sim.cfg.decision_s:
            plan[i] = plan.get(i, 0) + 1
        else:
            pri.pop(i)
    return plan


def planner(candidates, f1="true", budget=True, robust_q=90, window_s=30):
    """robust_q: plans are tried with OCR arrivals at this percentile of the last window_s seconds (None: this second).
    Fix 30/10: frozen at this second the planner missed 77 -> 112 new vehicles/s bursts that push D-FINE-X from 88 to
    241 ms; it kept choosing D-FINE-X and ran 17-26 % late."""
    past = {}
    est = {}
    reads = []

    def act(sim):
        reads.extend(getattr(sim, "reads_hist", []))
        del reads[:-window_s]
        now_reads = sim.ocr_crops * sum(s.cell().get("new", 0) for s in sim.streams)
        scale = (max(1.0, float(np.percentile(reads, robust_q)) / now_reads)
                 if robust_q is not None and reads and now_reads > 0 else 1.0)
        if f1 in ("probe_hist", "probe_sel"):
            update_estimates(sim, est)
        if sim.t < past.get("t", -1):  # a new episode on the same planner
            past.clear()
        if "prev" in past:  # what the Edges saw over the last decision (late share) + the F1 the planner believed
            for i, s in enumerate(sim.streams):
                t, late_sum, f1_sum = past["acc"][i]
                last = s.last or {"done": 1.0, "on_time": 1.0}
                past["acc"][i] = (t + 5, late_sum + 5 * (1 - last["done"] * last["on_time"]), f1_sum + 5 * past["prev"][i])
        past.setdefault("acc", [(0, 0.0, 0.0)] * len(sim.streams))
        best = None
        mode = "probe" if f1 in ("probe_hist", "probe_sel") else f1
        x = {k: v[0] for k, v in est.items()} if f1 in ("probe_hist", "probe_sel") else None
        for pol in candidates:
            acts = pol(frozen(sim, mode, x, scale))
            secs = frozen(sim, mode, x, scale).step(acts)
            val = episode_score(secs, past["acc"], episode_s=sim.cfg.episode_decisions * sim.cfg.decision_s) if budget else score(secs)
            if best is None or val > best[0]:
                best = (val, acts, [float(np.mean([x[i]["f1"] for x in secs])) for i in range(len(acts))])
        past["prev"], past["t"] = best[2], sim.t
        if f1 == "probe_sel":  # probe budget = what the fixed scheme spends (one probe-second per Edge stream)
            sim.probe_plan = plan_probes(sim, est, sum(m not in ("dfine_x_cloud",) for m, _ in best[1]))
        return best[1]
    return act


AGREE = "--agree" in sys.argv  # every policy (baselines too) sees the box-agreement probe instead of missed vehicles
AGREE_PATHS = ["results/detrac/probe_agree.jsonl", "results/vn_hard/probe_agree_vn.jsonl"]


def _init():
    o._init()
    if AGREE:
        o._G["train"] = o.TableV3.load(o.TABLE, "train", probe_paths=AGREE_PATHS, probe_field="probe_disagree")
        o._G["test"] = o.TableV3.load(o.TABLE, "test", probe_paths=AGREE_PATHS, probe_field="probe_disagree")
        o._G["profile"] = o.JCAB(o._G["train"]).profile
    o._G["prof"] = profiles(o._G["train"])


def task(args):
    d, dyn, static, theta, n, seed = args
    cfg = o.cfg_for("test", d, dyn)
    test = o._G["test"]
    cands = ([o.greedy(c) for c in o.CAPS] + [o.rescue(c, m) for c in o.CAPS for m in (o.DX, o.RT)]
             + [o.rules(*static), o.HeuristicThreshold(o._G["profile"], *theta), o.heuristic_net,
                o.rules(1.0, o.RT, 15), o.mpc])
    base = {"static": o.episode(test, cfg, n, seed, o.rules(*static)),
            "thresh": o.episode(test, cfg, n, seed, o.HeuristicThreshold(o._G["profile"], *theta)),
            "net": o.episode(test, cfg, n, seed, o.heuristic_net),
            "allrt": o.episode(test, cfg, n, seed, o.rules(1.0, o.RT, 15))}
    orc = [o.episode(test, cfg, n, seed, p) for p in [o.greedy(c) for c in o.CAPS]
           + [o.rescue(c, m) for c in o.CAPS for m in (o.DX, o.RT)]]
    sel = o.dataclasses.replace(cfg, probe_selective=True)
    now = {name: o.episode(test, sel if f1 == "probe_sel" else cfg, n, seed, planner(cands, f1, True))
           for name, f1 in (("true", "true"), ("probe", "probe"), ("probe_hist", "probe_hist"),
                            ("probe_sel", "probe_sel"))}
    return d, dyn, n, base, orc, now


def main():
    with mp.Pool(14, initializer=_init) as pool:
        tuned = dict(zip(CELLS, pool.starmap(o.tune, CELLS)))
        rows = pool.map(task, [(d, dyn, *tuned[(d, dyn)], n, s) for d, dyn in CELLS for n in NS for s in range(EPISODES)],
                        chunksize=1)
    for key in ("fps_goodput_c70", "ta200"):
        print(f"\n{key}: vs best baseline -- oracle (future) | planners without the future: true F1 / last probe /"
              " probe history / selective probing + history (same probe cost)")
        for d, dyn in CELLS:
            for n in NS:
                r = [x for x in rows if x[:3] == (d, dyn, n)]
                best = max(np.mean([x[3][p][key] for x in r]) for p in r[0][3])
                orc = np.mean([max([e[key] for e in x[4]] + [b[key] for b in x[3].values()]) for x in r])
                cols = "  ".join(f"{name} {(np.mean([x[5][name][key] for x in r]) / best - 1) * 100:+6.1f}%"
                                 for name in r[0][5])
                print(f"{str(d):16s} {dyn:4s} N{n:2d} best {best:7.2f}  oracle {(orc / best - 1) * 100:+6.1f}%  {cols}", flush=True)


if __name__ == "__main__":
    main()
