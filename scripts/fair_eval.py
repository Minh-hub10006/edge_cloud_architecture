"""Fair comparison (step 1, 30/10): a baseline is chosen per situation on the TRAIN videos, then run on test; 10 test
episodes per situation, mean +- std, and a paired sign-flip test (exact over all 2^10 sign patterns) of each planner
against that train-chosen baseline. The best baseline picked on test (the old, optimistic reference) is printed too.

Situations: OCR reads per vehicle x OCR workers x VN share x dynamics x N (as scripts/sim_v3_causal_bound.py).
Metric: constrained FPS, F1 >= .70, deadline 150 ms (fps_goodput_c70).

  PYTHONPATH=. python -B scripts/fair_eval.py [--agree] [--fixed-camera] [--probe-variants] [--drl DIR [--drl-plan]]
"""
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.sim_v3_causal_bound as cb  # noqa: E402  (adds --live: the testbed-fitted OCR model, c70 at 150 ms)

o = cb.o
if "--fixed-camera" in sys.argv:  # every policy sees fixed cameras (sim camera_mode "location"), tuning on train included
    _cfg_for = o.cfg_for
    o.cfg_for = lambda split, d, dyn: o.dataclasses.replace(_cfg_for(split, d, dyn), camera_mode="location")
KEY = "fps_goodput_c70"
NS, TRAIN_EPISODES, TEST_EPISODES = (10, 14), 5, 10
BASELINES = ("static", "thresh", "net", "allrt")
if "--singles" in sys.argv:  # round 29: every camera on one model at 15 fps, also eligible as the train-chosen baseline
    BASELINES += ("all_v8n", "all_y11s", "all_dx")
PLANNERS = (("true", "true"), ("probe", "probe"))
DRL_DIR = sys.argv[sys.argv.index("--drl") + 1] if "--drl" in sys.argv else None
if DRL_DIR:  # round 27: evaluate trained actors next to the realistic planner only
    PLANNERS = (("true", "true"), ("probe_sel", "probe_sel"))
DUMP = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
DRL_PLAN = "--drl-plan" in sys.argv  # round 29: DRL proposals tried on the frozen copy before acting (as the planner)
if "--no-planner" in sys.argv:  # round 30: the planner is left out of the comparison (user decision 1/11)
    PLANNERS = ()
if "--probe-variants" in sys.argv:  # step 2: how the probe is read (history, selective probing at the same GPU cost)
    PLANNERS = (("true", "true"), ("probe", "probe"), ("probe_hist", "probe_hist"), ("probe_sel", "probe_sel"))


def sign_flip_p(diffs):
    """Two-sided exact paired test: share of the 2^n sign patterns whose |mean| >= the observed |mean|."""
    d = np.asarray(diffs, dtype=float)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def baseline_policies(static, theta):
    return {"static": o.rules(*static), "thresh": o.HeuristicThreshold(o._G["profile"], *theta),
            "net": o.heuristic_net, "allrt": o.rules(1.0, o.RT, 15)}


def single_policies():
    return {name: (lambda sim, m=m: [(m, 15)] * len(sim.streams))
            for name, m in (("all_v8n", "yolo_edge"), ("all_y11s", "yolo11s_edge"), ("all_dx", "dfine_x_cloud"))}


def drl_policies():
    """{name: [policy per seed]}: each actor's greedy action per stream, and the same scores through the central
    allocator (drl/allocator.py); both set the selective probe plan like the training rollouts."""
    import glob
    import torch
    from drl import probe_plan
    from drl.allocator import allocate
    from drl.train_v3 import ACTIONS, net
    out = {"drl": [], "drl_alloc": []}
    for path in sorted(glob.glob(str(Path(DRL_DIR) / "actor_*.pt"))):
        ck = torch.load(path)
        actor = net(ck["dim"], len(ACTIONS))
        actor.load_state_dict(ck["actor"])
        for name in out:
            def act(sim, actor=actor, variant=ck["variant"], alloc=name == "drl_alloc"):
                with torch.no_grad():
                    logits = actor(torch.from_numpy(sim.observe(variant))).numpy()
                acts = allocate(sim, logits, ACTIONS) if alloc else [ACTIONS[i] for i in logits.argmax(-1)]
                sim.probe_plan = probe_plan.plan(sim, acts, probe_plan.edge_profile(sim.table))
                return acts
            out[name].append(act)
    return out


def drl_proposals(samples=1):
    """Policies proposing a full plan: each seed's greedy actor, plus `samples` per actor drawn from its softmax
    (a fixed random stream per proposal, so each decision gets a fresh but reproducible draw)."""
    import glob
    import torch
    from drl.train_v3 import ACTIONS, net
    out = []
    for k, path in enumerate(sorted(glob.glob(str(Path(DRL_DIR) / "actor_*.pt")))):
        ck = torch.load(path)
        actor = net(ck["dim"], len(ACTIONS))
        actor.load_state_dict(ck["actor"])
        for j in range(1 + samples):
            def act(sim, actor=actor, variant=ck["variant"], rng=np.random.default_rng(100 * k + j) if j else None):
                with torch.no_grad():
                    logits = actor(torch.from_numpy(sim.observe(variant))).numpy()
                if rng is None:
                    return [ACTIONS[i] for i in logits.argmax(-1)]
                p = np.exp(logits - logits.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                return [ACTIONS[rng.choice(len(ACTIONS), p=q)] for q in p]
            out.append(act)
    return out


def choose_on_train(args):
    """Mean c70 of every baseline on train episodes of this situation -> the name to use on test."""
    (d, dyn), static, theta, n = args
    cfg = o.cfg_for("train", d, dyn)
    pols = baseline_policies(static, theta) | single_policies()
    score = {b: float(np.mean([o.episode(o._G["train"], cfg, n, 1000 + s, pols[b])[KEY] for s in range(TRAIN_EPISODES)]))
             for b in BASELINES}
    return (d, dyn, n), max(score, key=score.get), score


def test_episode(args):
    (d, dyn), static, theta, n, seed = args
    cfg = o.cfg_for("test", d, dyn)
    pols = baseline_policies(static, theta)
    base = {b: o.episode(o._G["test"], cfg, n, seed, (pols | single_policies())[b])[KEY] for b in BASELINES}
    cands = ([o.greedy(c) for c in o.CAPS] + [o.rescue(c, m) for c in o.CAPS for m in (o.DX, o.RT)]
             + list(pols.values()))
    sel = o.dataclasses.replace(cfg, probe_selective=True)  # selective probing pays only for the frames it asks for
    plan = {name: o.episode(o._G["test"], sel if f1 == "probe_sel" else cfg, n, seed, cb.planner(cands, f1, True))[KEY]
            for name, f1 in PLANNERS}
    if DRL_PLAN:
        props = drl_proposals()
        plan["drl_plan"] = o.episode(o._G["test"], sel, n, seed, cb.planner(props, "probe_sel", True))[KEY]
        plan["drl_plan_mix"] = o.episode(o._G["test"], sel, n, seed, cb.planner(cands + props, "probe_sel", True))[KEY]
    if DRL_DIR:  # mean over the seeds' actors for this episode (paired with the baseline), per-seed values kept
        for name, pols_ in o._G.setdefault("drl", drl_policies()).items():
            vals = [o.episode(o._G["test"], sel, n, seed, p)[KEY] for p in pols_]
            plan[name] = float(np.mean(vals))
            plan[name + "_seeds"] = vals
    orc = max([o.episode(o._G["test"], cfg, n, seed, p)[KEY] for p in
               [o.greedy(c) for c in o.CAPS] + [o.rescue(c, m) for c in o.CAPS for m in (o.DX, o.RT)]] + list(base.values()))
    return (d, dyn, n), seed, base, plan, orc


def describe(d, dyn, n):
    reads, workers, vn = d
    return (f"{n} camera, {'phan lon' if vn > .5 else 'khoang nua'} VN, OCR {'nang' if reads > .5 else 'nhu testbed'} "
            f"{workers} worker, {'co' if dyn == 'all' else 'khong'} bien dong")


def main():
    cells = cb.CELLS
    with mp.Pool(14, initializer=cb._init) as pool:
        tuned = dict(zip(cells, pool.starmap(o.tune, cells)))
        chosen = dict((k, (b, sc)) for k, b, sc in pool.map(choose_on_train, [(c, *tuned[c], n) for c in cells for n in NS]))
        rows = pool.map(test_episode, [(c, *tuned[c], n, s) for c in cells for n in NS for s in range(TEST_EPISODES)],
                        chunksize=1)
    if DUMP:
        json.dump({"chosen": [[list(k), b, sc] for k, (b, sc) in chosen.items()],
                   "rows": [[list(k), seed, base, plan, orc] for k, seed, base, plan, orc in rows]}, open(DUMP, "w"))
    print(f"metric {KEY}; baseline chosen on train ({TRAIN_EPISODES} ep), test {TEST_EPISODES} ep; p = exact sign-flip test")
    summary = []
    for d, dyn in cells:
        for n in NS:
            r = sorted([x for x in rows if x[0] == (d, dyn, n)], key=lambda x: x[1])
            b, train_scores = chosen[(d, dyn, n)]
            base = np.array([x[2][b] for x in r])
            best_test = max(BASELINES, key=lambda k: np.mean([x[2][k] for x in r]))
            line = (f"{describe(d, dyn, n):60s} | baseline (train) {b:6s} {base.mean():6.1f}+-{base.std(ddof=1):4.1f} "
                    f"| best-on-test {best_test:6s} {np.mean([x[2][best_test] for x in r]):6.1f} "
                    f"| oracle {np.mean([x[4] for x in r]):6.1f}")
            for name in [p for p, _ in PLANNERS] + (["drl", "drl_alloc"] if DRL_DIR else []) + (["drl_plan", "drl_plan_mix"] if DRL_PLAN else []):
                v = np.array([x[3][name] for x in r])
                p = sign_flip_p(v - base)
                rel = (v.mean() / max(base.mean(), 1e-9) - 1) * 100
                line += f" | {name} {v.mean():6.1f}+-{v.std(ddof=1):4.1f} ({rel:+5.1f}%, p={p:.3f})"
                summary.append((name, describe(d, dyn, n), rel, p))
            print(line, flush=True)
    for name in [p for p, _ in PLANNERS] + (["drl", "drl_alloc"] if DRL_DIR else []) + (["drl_plan", "drl_plan_mix"] if DRL_PLAN else []):
        s = [x for x in summary if x[0] == name]
        print(f"\n{name}: significantly better (p<.05) in {sum(x[2] > 0 and x[3] < .05 for x in s)}/{len(s)}, "
              f"significantly worse in {sum(x[2] < 0 and x[3] < .05 for x in s)}/{len(s)}, median {np.median([x[2] for x in s]):+.1f}%")


if __name__ == "__main__":
    main()
