"""PPO for the v3 Cloud scheduler (docs/PLAN_DRL_V3B_VI.md P4) in drl/sim_v3.

  python -B -m drl.train_v3 --variant probe|none --seed 0 [--updates 300] [--out results/drl_v3]

One actor shared by every stream (no stream / Edge id): observation = SimV3.observe(variant) -> 12 actions
(model x FPS). Critic: the same observation (it already carries the shared GPU load, Cloud share and N).
Reward of stream i per 5 s decision: .5 x its TA@200 + .5 x mean TA@200 of all streams (a stream that grabs the GPU
pays for the others it delays) - .02 if it switched. "*_price" variants: TA@200 of the stream - GPU price x its GPU
share - .02 if it switched (the price, not the team term, makes streams split the GPU). "*_price2": price target .75
and - .5 x the stream's late/dropped share on top (the first price runs overloaded the GPU, 2-8% frames late).
"*_net": as *_price2, Edge sites on LAN / 4G / 5G (drl/net_traces.py), the site link in the state.
"*_net_dr": as *_net, trained with gpu_efficiency ~ U(.5, 1) per episode and each site's uplink x U(.4, 2).
"*_net_mm": as *_net_dr with Mahimahi LTE uplink traces (LAN 50% / 4G 50%), uplink x U(.5, 1.5) while training.
"*_lan_dyn": LAN only, dedicated Cloud; each episode (5 min) a random subset of the dynamics scenarios (surge, diurnal,
GPU thermal, Edge slowdown); state adds the GPU temperature and the Edge's current/nominal model time. Episodes: N uniform in [6, 30], 24 decisions, train split,
40% of streams on VN-Hard. Writes <out>/actor_<variant>_s<seed>.pt and a training log.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from drl import probe_plan
from drl.sim_v3 import FPS_LEVELS, MODES, SimV3, SimV3Config, TableV3

ACTIONS = list(itertools.product(MODES, FPS_LEVELS))
TABLE = "results/table_fps_all.jsonl"
PROBES = ("results/detrac/probe_table.jsonl", "results/vn_hard/probe_table_vn.jsonl")


def load_tables(split):
    return TableV3.load(TABLE, split, probe_paths=PROBES)


def net(inp, out):
    return nn.Sequential(nn.Linear(inp, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, out))


VARIANTS = ("probe", "none", "probe_price", "none_price", "probe_price2", "none_price2", "probe_net", "none_net",
            "probe_net_dr", "probe_net_mm", "none_net_mm", "probe_lan_dyn", "none_lan_dyn",
            "probe_lan_dyn_fps", "probe_net_mm_fps", "none_lan_dyn_fps", "none_net_mm_fps",
            "probe_lan_dyn_fps2", "probe_net_mm_fps2", "none_lan_dyn_fps2", "none_net_mm_fps2",
            "probe_lan_dyn_fps3", "probe_net_mm_fps3", "none_lan_dyn_fps3", "none_net_mm_fps3",
            "probe_net_5g_fps3", "none_net_5g_fps3", "probe_lan_fps3",
            "probe_net_5g", "probe_lan",  # TA reward (round 16, 29/10): metric TA@deadline, --deadline 200/150/100
            "probe_ocr_fps3", "none_ocr_fps3",  # round 18: LAN, Cloud shared with plate OCR (4090-measured load)
            "probe_ocr2_fps3",  # round 19: live plate OCR model (testbed 30/10), 4 or 8 OCR workers
            "probe_ocr2_fps3c70",  # round 19b: same, F1 constraint .70 (agreed 30/10: YOLO11s passes .60 on 85% of DETRAC)
            "probe_ocr3_fps3c70p",  # round 27: fixed cameras, selective probing, episode-projected reward, causal teacher
            "probe_ocr3_fps3c70s",  # round 28: same, reward edges inside the metric's hard limits (F1 .70-.75, late 3-5 %)
            "probe_ocr3_fps3c70r",  # round 30: reward of round 28, warm start from the train-tuned hand rule (no planner)
            "probe_ocr3_fps3c70t")  # round 31: as round 30 + half of each camera's reward shared by all cameras
# "*_fps" (27/10, objective agreed with the user): maximise on-time analytics FPS under per-camera constraints
FPS_F1_MIN, FPS_DROP_MAX = .60, .05
RULE_TABLE = "results/drl_r30/rule_table.json"
# drop penalty weight: "_fps" 4.0 (too cautious on 4G: kept almost everything on the Edge), "_fps2" 2.0
FPS_DROP_W = {"_fps": 4.0, "_fps2": 2.0, "_fps3": 0.0}
DYN_SCENARIOS = ("surge", "diurnal", "thermal", "edgeslow")
NET_MIX_MM = {"lan": .5, "4g": .5}  # Mahimahi: LTE only (no public saturated 5G uplink trace)
NET_MIX_5G = {"lan": .5, "5g": .5}  # measured 5G uplink (net_traces.load_5g_iperf), the paper's LAN+5G scenario
NET_MIX = {"lan": .5, "4g": .3, "5g": .2}  # docs/PLAN_DRL_V3_VI.md section 13


def config(variant, split="train", net=None, ul5g=None, gpu_eff=None, traces=None, rtt_ms=None, scenarios=None,
           episode_decisions=None, deadline_ms=None):
    """net: None = the variant's default (mobile links for *_net, LAN otherwise); True/False forces it.
    ul5g / gpu_eff: sensitivity overrides (5G uplink median Mbit/s, SimV3Config.gpu_efficiency)."""
    from drl import net_traces
    variant = variant.split("_fps")[0]  # same environment as the base variant; only the reward differs
    mobile = variant.endswith(("_net", "_net_dr", "_mm", "_5g")) if net is None else net
    traces = traces or ("mahimahi" if variant.endswith("_mm") else "5g" if variant.endswith("_5g") else "ucc")
    extra = {"gpu_efficiency": gpu_eff} if gpu_eff is not None else {}
    if variant.endswith("_dr") and split == "train":  # domain randomization while training only
        extra.update(gpu_efficiency_range=(.5, 1.0), uplink_scale_range=(.4, 2.0))
    if variant.endswith("_mm") and split == "train":  # milder uplink randomization around the unmodified traces
        extra.update(gpu_efficiency_range=(.5, 1.0), uplink_scale_range=(.5, 1.5))
    if variant.endswith("_ocr"):  # plate OCR on the Cloud GPU; training: crops/vehicle and dynamics random per episode
        extra.update(ocr_ms_per_crop=3.85)
        if split == "train":
            extra.update(ocr_crops_range=(0.0, 3.0), scenarios=DYN_SCENARIOS, scenario_random=True)
    if variant.endswith("_ocr2"):  # live OCR model fitted on the testbed; .45 reads/vehicle = the N=14 lots
        extra.update(ocr_read_ms=50.0, ocr_crops_per_vehicle=.45)
        extra.update(dynamics_keep_dataset=True)  # round 20 (30/10): surge/diurnal keep each stream's dataset
        if split == "train":  # reads/vehicle from none to 2x the testbed (better plate models read less), 4 or 8 workers,
            # VN share from the default .4 to the testbed's .7
            extra.update(ocr_crops_range=(0.0, .9), ocr_workers_choices=(4, 8), scenarios=DYN_SCENARIOS,
                         scenario_random=True, vn_share_range=(.4, .7))
    if variant.endswith("_ocr3"):  # round 27 (30/10): as _ocr2 + fixed cameras (one location each) + selective probing
        extra.update(ocr_read_ms=50.0, ocr_crops_per_vehicle=.45, dynamics_keep_dataset=True, camera_mode="location",
                     probe_selective=True)
        if split == "train":
            extra.update(ocr_crops_range=(0.0, .9), ocr_workers_choices=(4, 8), scenarios=DYN_SCENARIOS,
                         scenario_random=True, vn_share_range=(.4, .7))
    if variant.endswith(("_5g", "_lan", "_ocr", "_ocr2", "_ocr3")) and split == "train":  # around the fitted GPU efficiency 1.2 (sweep 29/10)
        extra.update(gpu_efficiency_range=(.8, 1.6), uplink_scale_range=(.5, 1.5))
    if variant.endswith("_dyn") and split == "train":  # LAN + dedicated Cloud: random subset of the dynamics, 5 min
        extra.update(scenarios=DYN_SCENARIOS, scenario_random=True, gpu_efficiency_range=(.6, 1.0), episode_decisions=60)
    if scenarios is not None:  # evaluation: exactly these scenarios
        extra.update(scenarios=tuple(scenarios), scenario_random=False)
    if episode_decisions is not None:
        extra["episode_decisions"] = episode_decisions
    if deadline_ms is not None:  # evaluation sensitivity: TA@150 / TA@100
        extra["deadline_ms"] = deadline_ms
    if mobile and traces == "mahimahi":
        extra.update(net_mix=NET_MIX_MM, net_traces=net_traces.load_mahimahi(split, rtt_ms=rtt_ms or 50.0))
    elif mobile and traces == "5g":
        extra.update(net_mix=NET_MIX_5G, net_traces=net_traces.load_5g_iperf(split, rtt_ms=rtt_ms or 60.0))
    elif mobile:
        extra.update(net_mix=NET_MIX, net_traces=net_traces.load(split, ul_median={"5g": ul5g} if ul5g else None))
    return SimV3Config(probe_period_s=5.0 if variant.startswith("probe") else 0.0,
                       price_target=.75 if variant.endswith(("_price2", "_net", "_dr", "_mm", "_dyn", "_5g", "_lan", "_ocr", "_ocr2", "_ocr3")) else .85, **extra)


def rollout(sim, actor, variant, n, rng, greedy=False, teacher=None):
    """One episode -> per-decision obs, actions, logp, rewards (arrays over streams) and the seconds.
    teacher: sim -> actions; when given it acts (behavior cloning data), the actor only scores its actions."""
    sim.reset(n)
    obs_l, act_l, logp_l, rew_l, secs = [], [], [], [], []
    f1_sum = np.zeros(n)  # per stream, F1 x seconds so far (the projected reward of "_fps3c70p" / "_fps3c70s")
    episode_s = sim.cfg.episode_decisions * sim.cfg.decision_s
    for _ in range(sim.cfg.episode_decisions):
        obs = sim.observe(variant)
        price = sim.price  # the price the streams decided on
        with torch.no_grad():
            logits = actor(torch.from_numpy(obs))
            dist = torch.distributions.Categorical(logits=logits)
            act = (torch.tensor([ACTIONS.index(x) for x in teacher(sim)]) if teacher is not None
                   else logits.argmax(-1) if greedy else dist.sample())
            logp = dist.log_prob(act)
        if sim.cfg.probe_selective and teacher is None:  # a teacher planner sets its own probe plan
            sim.probe_plan = probe_plan.plan(sim, [ACTIONS[a] for a in act.tolist()], probe_plan.edge_profile(sim.table))
        seconds = sim.step([ACTIONS[a] for a in act.tolist()])
        ta = np.array([np.mean([sec[i]["ta"] for sec in seconds]) for i in range(n)])
        switched = np.array([s.switched for s in sim.streams], dtype=float)
        obs_l.append(obs); act_l.append(act.numpy()); logp_l.append(logp.numpy())
        if "_fps" in variant:
            # per stream over the 5 s: on-time analytics FPS / 15, minus constraint violations (F1 >= .60,
            # dropped-or-late share <= 5 %), the Cloud price for its GPU share, and a switch cost
            use = np.array([np.mean([sec[i]["gpu_use"] for sec in seconds]) for i in range(n)])
            eff = np.array([np.mean([sec[i]["fps"] * sec[i]["done"] * sec[i]["on_time"] for sec in seconds]) for i in range(n)]) / 15
            f1 = np.array([np.mean([sec[i]["f1"] for sec in seconds]) for i in range(n)])
            drop = 1 - np.array([np.mean([sec[i]["done"] * sec[i]["on_time"] for sec in seconds]) for i in range(n)])
            if variant.endswith(("_fps3c70p", "_fps3c70s", "_fps3c70r", "_fps3c70t")):
                # round 27: the constraint as the metric scores it, F1 and late share projected to the episode end as
                # if this decision's rates were kept (so far + rate x time left), instead of this 5 s alone
                f1_sum += f1 * len(seconds)
                t = np.array([s.obs_t for s in sim.streams], dtype=float)
                left = np.maximum(len(seconds), episode_s - t)
                late_proj = (np.array([s.late_sum for s in sim.streams]) + drop * left) / (t + left)
                f1_proj = (f1_sum + f1 * left) / (t + left)
                if variant.endswith(("_fps3c70s", "_fps3c70r", "_fps3c70t")):
                    # round 28 (31/10): round 27 sat at 4-6 % late and skipped RT-DETR because the soft edges paid
                    # up to 10 % late / F1 .65 while the metric cuts at 5 % / .70; full credit only with a margin
                    ok = np.clip((f1_proj - .70) / .05, 0, 1) * np.clip((FPS_DROP_MAX - late_proj) / .02, 0, 1)
                else:
                    ok = np.clip((f1_proj - .65) / .10, 0, 1) * np.clip((2 * FPS_DROP_MAX - late_proj) / FPS_DROP_MAX, 0, 1)
                own = eff * ok
                if variant.endswith("_fps3c70t"):
                    # round 31 (1/11): round 30 sent cameras to D-FINE-X (own F1 up) and made the others late (22 %);
                    # half of the reward is the mean over all cameras, so a camera pays for the delay it causes
                    own = .5 * own + .5 * own.mean()
                rew_l.append(own - price * use - .02 * switched)
            elif variant.endswith(("_fps3", "_fps3c70")):
                f1_min = .70 if variant.endswith("c70") else FPS_F1_MIN
                # the metric itself: FPS only counts while the constraints hold (soft edges .55-.65 F1, 5-10 % drop)
                # -- _fps/_fps2 still paid FPS to constraint-violating streams and collapsed to YOLO11s@15 on 4G
                ok = np.clip((f1 - (f1_min - .05)) / .10, 0, 1) * np.clip((2 * FPS_DROP_MAX - drop) / FPS_DROP_MAX, 0, 1)
                rew_l.append(eff * ok - price * use - .02 * switched)
            else:
                rew_l.append(eff - 2.0 * np.maximum(0, FPS_F1_MIN - f1) / FPS_F1_MIN
                             - FPS_DROP_W["_fps" + variant.split("_fps")[1]] * np.maximum(0, drop - FPS_DROP_MAX)
                             - price * use - .02 * switched)
        elif variant.endswith(("_price", "_price2", "_net", "_dr", "_mm", "_dyn", "_5g", "_lan")):
            use = np.array([np.mean([sec[i]["gpu_use"] for sec in seconds]) for i in range(n)])
            late = np.array([1 - np.mean([sec[i]["done"] * sec[i]["on_time"] for sec in seconds]) for i in range(n)])
            rew_l.append(ta - price * use - .02 * switched - (.5 * late if variant.endswith(("_price2", "_net", "_dr", "_mm", "_dyn", "_5g", "_lan")) else 0.0))
        else:
            rew_l.append(.5 * ta + .5 * ta.mean() - .02 * switched)
        secs += seconds
    return np.stack(obs_l), np.stack(act_l), np.stack(logp_l), np.stack(rew_l), secs


def gae(rew, val, gamma=.95, lam=.95):
    adv, last = np.zeros_like(rew), np.zeros(rew.shape[1])
    for t in reversed(range(len(rew))):
        nxt = val[t + 1] if t + 1 < len(val) else np.zeros(rew.shape[1])
        delta = rew[t] + gamma * nxt - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def behavior_clone(actor, critic, table, cfg, variant, dim, rng, episodes, epochs, n_min, n_max):
    """Warm start (round 20): the greedy oracle (next 5 s known, GPU cap .9) acts; the actor learns its actions from
    the observation alone (cross-entropy), the critic its discounted returns under the same reward."""
    if variant.endswith(("_fps3c70r", "_fps3c70t")):  # round 30: the hand rule tuned on train for the nearest situation (scripts/rule_table.py)
        from scripts.rule_table import lookup
        from scripts.sim_v3_baselines import rules
        rule_tab = json.load(open(RULE_TABLE))

        def teacher(sim):  # teacher only: like the tuned baseline it knows the situation (VN share, OCR, workers)
            vn = float(np.mean([s.key[0] != "detrac" for s in sim.streams]))
            share, mode, fps = lookup(rule_tab, sim.ocr_crops, sim.ocr_workers, vn)
            return rules(share, mode, fps)(sim)
    elif variant.split("_fps")[0].endswith("_ocr3"):  # round 27: the causal planner of round 26 (no future, robust OCR)
        import scripts.sim_v3_causal_bound as cb
        o = cb.o
        if "prof" not in o._G:
            cb._init()
        cands = ([o.greedy(c) for c in o.CAPS] + [o.rescue(c, m) for c in o.CAPS for m in (o.DX, o.RT)]
                 + [o.rules(4, o.DX, 15), o.rules(1.0, o.RT, 15), o.heuristic_net,
                    o.HeuristicThreshold(o._G["profile"], .02, .9)])
        planners = {}
        teacher = lambda sim: planners.setdefault(id(sim), cb.planner(cands, "probe_sel", True))(sim)  # noqa: E731
    else:
        from scripts.sim_v3_oracle_shared import greedy
        teacher = greedy(.9)
    data = []
    for _ in range(episodes):
        o, a, _lp, r, _ = rollout(SimV3(table, cfg, rng.random()), actor, variant, rng.randint(n_min, n_max), rng,
                                  teacher=teacher)
        _adv, ret = gae(r, np.zeros_like(r), lam=1.0)
        data.append((o.reshape(-1, dim), a.reshape(-1), ret.reshape(-1)))
    O, A, RET = (torch.from_numpy(np.concatenate([d[k] for d in data])) for k in range(3))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    for epoch in range(epochs):
        for idx in torch.randperm(len(O)).split(2048):
            ce = nn.functional.cross_entropy(actor(O[idx]), A[idx].long())
            vl = ((critic(O[idx]).squeeze(-1) - RET[idx].float()) ** 2).mean()
            opt.zero_grad(); (ce + .5 * vl).backward(); opt.step()
    with torch.no_grad():
        acc = float((actor(O).argmax(-1) == A).float().mean())
    return {"bc_samples": len(O), "bc_accuracy": acc}


def train(variant, seed, updates, episodes, out, n_min=6, n_max=30, deadline=None, bc_episodes=0, bc_epochs=30,
          lr=3e-4):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    table = load_tables("train")
    cfg = config(variant, deadline_ms=deadline)
    tag = f"{variant}_d{deadline:.0f}" if deadline is not None else variant
    probe_sim = SimV3(table, cfg, 0)
    probe_sim.reset(6)
    dim = probe_sim.observe(variant).shape[1]
    actor, critic = net(dim, len(ACTIONS)), net(dim, 1)
    log = []
    if bc_episodes:
        log.append(behavior_clone(actor, critic, table, cfg, variant, dim, rng, bc_episodes, bc_epochs, n_min, n_max))
        print(f"[{tag} s{seed}] behavior cloning {log[-1]}", flush=True)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    for update in range(updates):
        batch = []
        for _ in range(episodes):
            n = rng.randint(n_min, n_max)
            o, a, lp, r, _secs = rollout(SimV3(table, cfg, rng.random()), actor, variant, n, rng)
            with torch.no_grad():
                v = critic(torch.from_numpy(o.reshape(-1, dim))).numpy().reshape(r.shape)
            adv, ret = gae(r, v)
            batch.append((o.reshape(-1, dim), a.reshape(-1), lp.reshape(-1), adv.reshape(-1), ret.reshape(-1), r.mean()))
        O, A, LP, ADV, RET = (torch.from_numpy(np.concatenate([b[k] for b in batch])) for k in range(5))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(O)).split(2048):
                dist = torch.distributions.Categorical(logits=actor(O[idx]))
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(.8, 1.2) * ADV[idx]).mean()
                vl = ((critic(O[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                loss = pg + .5 * vl - .01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), .5); opt.step()
        log.append({"update": update, "reward": float(np.mean([b[5] for b in batch]))})
        if update == 0 and bc_episodes:
            print(f"[{tag} s{seed}] reward after behavior cloning {log[-1]['reward']:.3f}", flush=True)
        if update % 20 == 0 or update == updates - 1:
            print(f"[{tag} s{seed}] update {update} reward {log[-1]['reward']:.3f}", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "dim": dim, "variant": variant, "actions": ACTIONS, "deadline": deadline},
               out / f"actor_{tag}_s{seed}.pt")
    (out / f"train_{tag}_s{seed}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log), encoding="utf-8")


def load_policy(path):
    """Greedy actor for scripts/sim_v3_baselines.py: sim -> actions."""
    ck = torch.load(path)
    actor = net(ck["dim"], len(ACTIONS))
    actor.load_state_dict(ck["actor"])
    variant = ck["variant"]

    def act(sim):
        with torch.no_grad():
            idx = actor(torch.from_numpy(sim.observe(variant))).argmax(-1).tolist()
        acts = [ACTIONS[i] for i in idx]
        if sim.cfg.probe_selective:
            sim.probe_plan = probe_plan.plan(sim, acts, probe_plan.edge_profile(sim.table))
        return acts
    return act, variant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, default="probe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--out", default="results/drl_v3")
    p.add_argument("--n-min", type=int, default=6, help="smallest N per training episode")
    p.add_argument("--n-max", type=int, default=30, help="largest N per training episode")
    p.add_argument("--deadline", type=float, default=None, help="deadline ms of TA / on-time (default 200)")
    p.add_argument("--bc-episodes", type=int, default=0, help="behavior cloning warm start from the greedy oracle")
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate (1e-4 after a warm start)")
    a = p.parse_args()
    train(a.variant, a.seed, a.updates, a.episodes, a.out, a.n_min, a.n_max, a.deadline, a.bc_episodes, a.bc_epochs,
          a.lr)


if __name__ == "__main__":
    main()
