"""MAPPO (shared actor, centralized critic) on drl.sim.MultiEdgeSim -> edge-drl-actor-v2 JSON.

  python -B -m drl.mappo train --table T1.jsonl ... --scene scene.jsonl --out actor.json [--iters 300]
  python -B -m drl.mappo eval  --table ... --scene ... [--actor actor.json]   # vs fixed/rule baselines

Actor: OBS_V2 (22) -> 64 -> 64 -> 4 (tanh), parameter-shared by every Edge, masked softmax.
Critic: each Edge's observation + simulator global state (Edge count, Cloud headroom/load, outage) -> that
Edge's value (centralized information, agent-specific value). Advantage per Edge from its own reward plus the
shared fairness term: with a single team advantage, one Edge's good Cloud choice drowned in the others' noise
and every seed collapsed to Edge-only YOLO11s (2026-10-13). GAE; PPO clip; lambda fixed (sweep) or by dual
ascent on the mean F1 shortfall below A_min (constraint: shortfall <= tolerance).
Needs torch (training host only); the Edge runs the exported numpy actor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drl.sim import CLOUD_MODES, REAL_EDGES, MultiEdgeSim, QualityTable, SimConfig, extra_feature_names, jain
from edge_agent.drl_policy import ACTION_MODES, OBS_V2, MlpActor, write_actor

HIDDEN = 64
# Eight Edges (real profiles with jitter + synthetic devices) on one GPU, LAN; 70-100% of the GPU for us.
SCALE8 = dict(edges=(8, 8), network_ms=(3, 6), wan_p=0, cloud_background=(.7, 1))
# Beyond ~11 Edges even RT-DETR (172.5 FPS) cannot serve every stream: some Edges must stay local.
SCALE12 = dict(SCALE8, edges=(12, 12))
SCALE16 = dict(SCALE8, edges=(16, 16))
# Within-episode dynamics: GPU share redrawn about once a minute, encoder swings and network spells.
DYNAMIC = dict(dyn_bg_p=1 / 60, dyn_enc_p=1 / 120, dyn_net_p=1 / 90)
DYN12 = dict(edges=(12, 12), network_ms=(3, 6), wan_p=.2, cloud_background=(.4, 1), **DYNAMIC)
DYN8 = dict(DYN12, edges=(8, 8))
# 4-12 cameras on the measured testbed plus external conditions: other GPU tenants (40-100% left), Edge CPU
# contention (local models 50-100% speed), encoder swings, network spells (LAN 3-6 ms, sometimes WAN).
EXTERNAL = dict(network_ms=(3, 6), wan_p=.1, cloud_background=(.5, 1), dyn_edge_p=1 / 120, **DYNAMIC)
N4_12 = dict(edges=(4, 12), **EXTERNAL)
N_EVAL = {n: dict(EXTERNAL, edges=(n, n)) for n in (4, 6, 8, 10, 12)}
# The measured testbed at n cameras: real Edge profiles (+-20% jitter), LAN, the whole GPU, no disturbances.
N_TESTBED = {n: dict(edges=(n, n), network_ms=(3, 6), wan_p=0, cloud_background=(1, 1), random_profile_p=0)
             for n in (4, 6, 8, 10, 12)}
# Diverse but realistic deployment (no GPU tenants, fixed camera count per episode): each camera keeps a scene
# regime (night / rain / sun / cloud); 20% of Edges behind WAN/4G links (20-80 ms, redrawn ~every 90 s); Edge CPU
# contention (local models 50-100% speed, ~every 2 min) and slow encoder drift (~every 10 min); Cloud overload
# comes from the cameras' own choices. Assumption-based except the regimes (UA-DETRAC weather tags).
DIVERSE = dict(edges=(4, 12), network_ms=(3, 6), wan_p=0, wan_ms=(20, 80), edge_wan_p=.2, cloud_background=(1, 1),
               scene_regime=True, dyn_net_p=1 / 90, dyn_edge_p=1 / 120, dyn_enc_p=1 / 600)
N_DIVERSE = {n: dict(DIVERSE, edges=(n, n)) for n in (4, 6, 8, 10, 12)}
# Simple accuracy/FPS heuristic: most accurate first, one step down the ladder when results come late, one step up
# when they are all on time. Ladder: LW-DETR 15 -> 10 -> 5 FPS on the Cloud -> YOLO11s 15 FPS on the Edge.
LADDER = (("lwdetr_large_cloud", 15), ("lwdetr_large_cloud", 10), ("lwdetr_large_cloud", 5), ("yolo11s_edge", 15))


SIM_FORMAT = "sim-drl-actor-v2p"  # OBS_V2 + priority: simulator experiments only, the Edge rejects it


def _nets(torch, global_dim, obs_dim=len(OBS_V2), n_actions=len(ACTION_MODES)):
    nn = torch.nn
    actor = nn.Sequential(nn.Linear(obs_dim, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                          nn.Linear(HIDDEN, n_actions))
    critic = nn.Sequential(nn.Linear(obs_dim + global_dim, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN),
                           nn.Tanh(), nn.Linear(HIDDEN, 1))
    return actor, critic


def export(actor, path, extra=("priority",), actions=None):
    """torch Linear stores (out, in); the Edge runtime wants (in, out) with the same activations."""
    linears = [m for m in actor if hasattr(m, "weight")]
    acts = ["tanh"] * (len(linears) - 1) + ["linear"]
    layers = [(m.weight.detach().numpy().T, m.bias.detach().numpy(), a) for m, a in zip(linears, acts)]
    if layers[0][0].shape[0] == len(OBS_V2) and not actions:
        write_actor(path, ACTION_MODES, layers)
        return
    Path(path).write_text(json.dumps({"format": SIM_FORMAT, "observation": [*OBS_V2, *extra],
                                      "actions": list(actions or ACTION_MODES),
                                      "layers": [{"w": w.tolist(), "b": b.tolist(), "act": a} for w, b, a in layers]}),
                          encoding="utf-8")


class SimActor(MlpActor):
    """Numpy actor over OBS_V2 + priority (simulator-only format)."""

    def __init__(self, spec):
        if spec.get("format") != SIM_FORMAT:
            raise ValueError(f"simulator actor format must be {SIM_FORMAT}")
        self.actions = tuple(spec["actions"])
        acts = {"tanh": np.tanh, "linear": lambda x: x}
        self.layers = [(np.asarray(l["w"]), np.asarray(l["b"]), acts[l["act"]]) for l in spec["layers"]]


def load_actor(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimActor(spec) if spec.get("format") == SIM_FORMAT else MlpActor(spec)


def collect(sim, actor, critic, torch, episodes, team_mix=0.0):
    """-> list of episode dicts: per-decision obs, masks, actions, logp, critic input, team reward, info."""
    out = []
    for _ in range(episodes):
        obs, steps, done = sim.reset(), [], False
        while not done:
            masks = np.array(sim.masks())
            crit = np.concatenate([obs, np.tile(sim.global_state(), (len(obs), 1))], axis=1)
            with torch.no_grad():
                logits = actor(torch.as_tensor(obs, dtype=torch.float32))
                logits = logits.masked_fill(~torch.as_tensor(masks), -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)
                value = critic(torch.as_tensor(crit, dtype=torch.float32)).squeeze(-1).numpy()
            next_obs, reward, done, info = sim.step(actions.tolist())
            own = np.array(info["rewards"])
            steps.append({"obs": obs, "masks": masks, "actions": actions.numpy(), "logp": logp.numpy(),
                          "crit": crit, "value": value, "reward": reward,
                          # own reward (team_mix of it replaced by the Edges' mean, so an Edge feels the GPU overload
                          # it causes the others) + shared fairness term
                          "rewards": (1 - team_mix) * own + team_mix * own.mean() + (reward - own.mean()),
                          "info": info})
            obs = next_obs
        out.append(steps)
    return out


def gae(steps, gamma=.99, lam=.95):
    """Per-Edge GAE -> (advantages, returns), each (decisions, Edges)."""
    rewards = np.array([s["rewards"] for s in steps])
    values = np.array([s["value"] for s in steps])
    advantages, last = np.zeros_like(rewards), np.zeros(rewards.shape[1])
    for t in reversed(range(len(steps))):
        next_value = values[t + 1] if t + 1 < len(steps) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    return advantages, advantages + values


def train(table, out, *, iters=300, episodes=16, epochs=8, lr=3e-4, clip=.2, entropy=.02, lagrange_lr=5.0, tolerance=.005,
          cfg=None, seed=0, log=print, eval_table=None, select_cfg=None, curriculum=False, team_mix=0.0):
    import dataclasses
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)  # several seeds train side by side
    cfg = cfg or SimConfig()
    # Checkpoint selection on the measured testbed (validation sequences), at the initial, fixed lambda.
    eval_cfg = dataclasses.replace(select_cfg or dataclasses.replace(
        cfg, real_edges_only=True, network_ms=(3, 3), wan_p=0, cloud_background=(1, 1)), lagrange=cfg.lagrange)
    sim = MultiEdgeSim(table, cfg, seed=seed)
    sim.reset()
    actor, critic = _nets(torch, len(sim.global_state()), sim.observe().shape[1], len(sim.actions))
    names = [m if len(cfg.fps_levels) == 1 else f"{m}@{f}" for m, f in sim.actions]
    multi = names if len(cfg.fps_levels) > 1 else None
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    best = None
    top = cfg.edges[1]
    for it in range(iters):
        if curriculum:
            # Grow the Edge count from 2-4 to the full range over the first 60% of training: with many Edges
            # exploring at once the Cloud saturates and Cloud actions get punished before any can pay off.
            frac = min(1.0, it / (.6 * iters))
            cfg.edges = (2, max(4, round(4 + (top - 4) * frac)))
        batch = collect(sim, actor, critic, torch, episodes, team_mix)
        obs, masks, acts, old_logp, adv_rows, crit, ret = [], [], [], [], [], [], []
        f1s = []
        for steps in batch:
            advantages, returns = gae(steps)
            for s, a, r in zip(steps, advantages, returns):
                # One actor row and one critic row per Edge per decision.
                obs.append(s["obs"]); masks.append(s["masks"]); acts.append(s["actions"]); old_logp.append(s["logp"])
                adv_rows.append(a); crit.append(s["crit"]); ret.append(r)
                f1s.extend(e["f1"] for e in s["info"]["edges"])
        obs_t = torch.as_tensor(np.concatenate(obs), dtype=torch.float32)
        mask_t = torch.as_tensor(np.concatenate(masks), dtype=torch.bool)
        act_t = torch.as_tensor(np.concatenate(acts), dtype=torch.long)
        old_t = torch.as_tensor(np.concatenate(old_logp), dtype=torch.float32)
        adv_t = torch.as_tensor(np.concatenate(adv_rows), dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        crit_t = torch.as_tensor(np.concatenate(crit), dtype=torch.float32)
        ret_t = torch.as_tensor(np.concatenate(ret), dtype=torch.float32)
        for _ in range(epochs):
            for idx in torch.randperm(len(obs_t)).split(512):
                logits = actor(obs_t[idx]).masked_fill(~mask_t[idx], -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act_t[idx]) - old_t[idx])
                a = adv_t[idx]
                loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean() - entropy * dist.entropy().mean()
                actor_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
                actor_opt.step()
            for idx in torch.randperm(len(crit_t)).split(256):
                value_loss = ((critic(crit_t[idx]).squeeze(-1) - ret_t[idx]) ** 2).mean()
                critic_opt.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
                critic_opt.step()
        mean_f1 = float(np.mean(f1s))
        # Constraint on the same quantity the reward penalizes: mean per-decision F1 shortfall <= tolerance.
        deficit = float(np.mean([max(0.0, cfg.a_min - f) for f in f1s]))
        cfg.lagrange = max(0.0, cfg.lagrange + lagrange_lr * (deficit - tolerance))
        team = float(np.mean([sum(s["reward"] for s in steps) for steps in batch]))
        modes = np.bincount(np.concatenate(acts), minlength=len(names)) / len(np.concatenate(acts))
        log(json.dumps({"iter": it, "team_return": round(team, 3), "mean_f1": round(mean_f1, 4), "f1_deficit": round(deficit, 4),
                        "lagrange": round(cfg.lagrange, 3),
                        "mode_share": {m: round(float(x), 3) for m, x in zip(names, modes) if x > 0}}))
        if eval_table is not None and (it + 1) % 20 == 0:
            export(actor, out, extra_feature_names(cfg), multi)
            result = evaluate(eval_table, load_actor(out), cfg=eval_cfg, seed=1234)
            # Feasible checkpoints (accuracy constraint met) beat infeasible ones; then the higher return.
            score = (result["f1_deficit"] <= tolerance, result["team_return"])
            if best is None or score > best[0]:
                best = (score, [p.detach().clone() for p in actor.parameters()])
            log(json.dumps({"iter": it, "eval_team_return": round(result["team_return"], 3),
                            "eval_f1_deficit": round(result["f1_deficit"], 4)}))
    if best is not None:
        with torch.no_grad():
            for p, b in zip(actor.parameters(), best[1]):
                p.copy_(b)
    export(actor, out, extra_feature_names(cfg), multi)
    return actor


# ---- evaluation and baselines (numpy only) ----
BASELINES = [f"always_{m}" for m in ACTION_MODES] + ["rules", "rules_acc", "best_fixed", "oracle"]


def _rules_fps(e, timely, mask, sim, up=.97, down=.9):
    """LADDER heuristic for one Edge: e = its sim Edge (current mode/FPS), timely = last window's timely ratio."""
    k = LADDER.index((e.mode, e.fps)) if (e.mode, e.fps) in LADDER else 0
    k = min(k + 1, len(LADDER) - 1) if timely < down else max(k - 1, 0) if timely >= up else k
    while not mask[sim.action_index(*LADDER[k])] and k < len(LADDER) - 1:
        k += 1  # Cloud masked (stale feedback): fall to the Edge
    return sim.action_index(*LADDER[k])


def _rules_acc(obs, mask):
    """Accuracy-aware heuristic from live observations only: Cloud LW-DETR when the Edge publishes fast enough
    and the Cloud has room, step down to RT-DETR then YOLO11s on the Edge when results come late."""
    o = dict(zip(OBS_V2, obs))
    current = max(ACTION_MODES, key=lambda m: o[f"mode_{m}"])
    lw, rt, y11 = (ACTION_MODES.index(m) for m in ("lwdetr_large_cloud", "rtdetr_cloud", "yolo11s_edge"))
    weak_encoder = o["cap_encode"] * 300 < 30  # x264 cannot publish 15 FPS while YOLO runs (252: 20.9)
    if weak_encoder or not mask[lw] or o["cloud_pressure"] >= .6:
        return y11
    if current == "lwdetr_large_cloud" and o["timely_ratio"] < .8:
        return rt
    if current == "rtdetr_cloud":
        return y11 if o["timely_ratio"] < .8 else rt
    return lw


def _policy(name, actor=None):
    fixed = {f"always_{m}": ACTION_MODES.index(m) for m in ACTION_MODES}
    if name in fixed:
        return lambda obs, mask: fixed[name] if mask[fixed[name]] else 0
    if name == "rules_acc":
        return _rules_acc
    if name == "rules":
        # Cloud RT-DETR while it keeps up; back to YOLOv8n on the Edge when late, backlogged or masked.
        late, backlog = OBS_V2.index("timely_ratio"), OBS_V2.index("edge_backlog")
        def rules(obs, mask):
            ok = obs[late] > .9 and obs[backlog] < .3 and mask[ACTION_MODES.index("rtdetr_cloud")]
            return ACTION_MODES.index("rtdetr_cloud") if ok else 0
        return rules
    if name == "actor":
        return actor.act
    raise ValueError(name)


def best_fixed_modes(table, cfg, seed=1234, episodes=10):
    """Per device: the single mode with the best timely ratio minus 5 x mean F1 shortfall (baseline that knows
    the device but never adapts)."""
    score = {}
    for m in ACTION_MODES:
        for p in evaluate(table, policy=f"always_{m}", cfg=cfg, seed=seed, episodes=episodes, raw=True):
            score.setdefault(p["edge"], {}).setdefault(m, []).append(p["timely"] - 5 * max(0.0, cfg.a_min - p["f1"]))
    return {edge: max(by_mode, key=lambda m: np.mean(by_mode[m])) for edge, by_mode in score.items()}


def _metrics(parts, cfg):
    mean = lambda k, xs=parts: float(np.mean([p[k] for p in xs]))  # noqa: E731
    return {"timely": mean("timely"), "f1": mean("f1"),
            "below_a_min": float(np.mean([p["f1"] < cfg.a_min for p in parts])),
            "f1_deficit": float(np.mean([max(0.0, cfg.a_min - p["f1"]) for p in parts])),
            "latency_ms": mean("latency"), "latency_p95_ms": float(np.percentile([p["latency"] for p in parts], 95)),
            "loss": mean("loss"), "switch_rate": mean("switch"),
            "cloud_share": float(np.mean([p["mode"] in CLOUD_MODES for p in parts])),
            **{f"ta_{d}": float(np.mean([p["ta"][d] for p in parts])) for d in parts[0]["ta"]}}


def evaluate(table, actor=None, *, policy="actor", cfg=None, seed=1234, episodes=20, raw=False):
    cfg = cfg or SimConfig()
    sim = MultiEdgeSim(table, cfg, seed=seed)
    fixed_by_edge = best_fixed_modes(table, cfg, seed, episodes) if policy == "best_fixed" else None
    act = None if policy in ("oracle", "best_fixed", "rules_fps") else _policy(policy, actor)
    returns, parts, fairness = [], [], []
    for _ in range(episodes):
        obs, done, total = sim.reset(), False, 0.0
        while not done:
            masks = sim.masks()
            if policy == "oracle":
                actions = sim.oracle_actions()
            elif policy == "best_fixed":
                # Synthetic devices (randomized scenarios) take the choice of the real Edge with the nearest encoder.
                real = {p.name: p.encode_fps for p in REAL_EDGES}
                actions = []
                for e, m in zip(sim.edges, masks):
                    name = e.p.name if e.p.name in fixed_by_edge else min(real, key=lambda n: abs(real[n] - e.p.encode_fps))
                    a = sim.action_index(fixed_by_edge.get(name, "yolo11s_edge"))
                    actions.append(a if m[a] else sim.action_index("yolo_edge"))
            elif policy == "actor":
                actions = [act(o, m) for o, m in zip(obs, masks)]
            elif policy == "rules_fps":
                t = OBS_V2.index("timely_ratio")
                actions = [_rules_fps(e, o[t], m, sim) for e, o, m in zip(sim.edges, obs, masks)]
            else:  # baselines decide among ACTION_MODES at the default 15 FPS
                base_masks = [[m[sim.action_index(mode)] for mode in ACTION_MODES] for m in masks]
                actions = [sim.action_index(ACTION_MODES[act(o, m)]) for o, m in zip(obs, base_masks)]
            obs, reward, done, info = sim.step(actions)
            total += reward
            parts.extend(info["edges"])
            fairness.append(jain([p["timely"] for p in info["edges"]]))
        returns.append(total)
    if raw:
        return parts
    per_edge = {}
    for p in parts:
        per_edge.setdefault(p["edge"], []).append(p)
    return {"policy": policy, "team_return": float(np.mean(returns)), **_metrics(parts, cfg),
            "jain": float(np.mean(fairness)),
            "per_edge": {e: {k: round(v, 4) for k, v in _metrics(ps, cfg).items()} for e, ps in sorted(per_edge.items())}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("train", "eval"))
    p.add_argument("--table", nargs="+", required=True)
    p.add_argument("--scene")
    p.add_argument("--out", default="actor.json")
    p.add_argument("--actor", nargs="*", default=[], help="one or more actor JSONs (e.g. one per seed)")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--a-min", type=float, default=SimConfig.a_min)
    p.add_argument("--seed", type=int, default=0)
    # Allowed mean F1 shortfall below A_min; must exceed the oracle's shortfall or lambda grows without bound.
    p.add_argument("--tolerance", type=float, default=.005)
    # Fixed-lambda sweep: --lagrange-lr 0 --lagrange L (dual ascent overshoots when the floor is barely feasible).
    p.add_argument("--lagrange", type=float, default=SimConfig.lagrange)
    p.add_argument("--lagrange-lr", type=float, default=5.0)
    # testbed: the measured deployment (2-4 Edges, LAN 3-6 ms, 80-100% of the GPU) with the usual device jitter and
    # synthetic devices. With one Cloud request in flight per Edge, Cloud throughput is 1000/RTT and the LAN round
    # trip (~67 ms) sits right at 15 FPS, so a WAN-heavy domain makes Cloud a loss on every device.
    # contention: many Edges share one GPU (2-10; 8 x 15 FPS exceeds LW-DETR's 85.7 FPS), where always-Cloud
    # overloads it and the Edges must split Cloud and local work; checkpoints selected on the 8-Edge scenario.
    # large: 4-16 Edges, more than the GPU can serve on the Cloud; selection on the 12-Edge scenario.
    # dynamic: large + within-episode GPU share / encoder / network changes (measured ranges).
    p.add_argument("--domain", choices=("randomized", "testbed", "contention", "large", "dynamic", "n4_12"),
                   default="randomized")
    p.add_argument("--priority", action="store_true", help="simulator-only per-Edge priority feature")
    p.add_argument("--coord", action="store_true", help="simulator-only Cloud-share and GPU-busy features")
    p.add_argument("--async-share", type=float, default=1.0, help="share of Edges that may switch per decision")
    p.add_argument("--fps-levels", type=int, nargs="+", default=[15], help="analytics FPS the policy may choose")
    p.add_argument("--curriculum", action="store_true", help="grow the Edge count during training")
    p.add_argument("--team-mix", type=float, default=0.0, help="share of each Edge's reward replaced by the team mean")
    p.add_argument("--reward", choices=("constrained", "ta"), default="constrained")
    p.add_argument("--ta-deadline", type=float, default=200.0)
    args = p.parse_args(argv)
    common = dict(reward=args.reward, ta_deadline_ms=args.ta_deadline, priority_feature=args.priority,
                  coord_features=args.coord, async_share=args.async_share, fps_levels=tuple(args.fps_levels))
    cfg = SimConfig(a_min=args.a_min, lagrange=args.lagrange, **common)
    select = None
    if args.domain == "contention":
        cfg = SimConfig(a_min=args.a_min, lagrange=args.lagrange, edges=(2, 10), network_ms=(3, 6), wan_p=.1,
                        cloud_background=(.5, 1), **common)
        select = SimConfig(a_min=args.a_min, lagrange=args.lagrange, **SCALE8, **common)
    if args.domain == "dynamic":
        cfg = SimConfig(a_min=args.a_min, lagrange=args.lagrange, edges=(4, 16), network_ms=(3, 6), wan_p=.2,
                        cloud_background=(.4, 1), **DYNAMIC, **common)
        select = SimConfig(a_min=args.a_min, lagrange=args.lagrange, **DYN12, **common)
    if args.domain == "large":
        cfg = SimConfig(a_min=args.a_min, lagrange=args.lagrange, edges=(4, 16), network_ms=(3, 6), wan_p=.1,
                        cloud_background=(.6, 1), **common)
        select = SimConfig(a_min=args.a_min, lagrange=args.lagrange, **SCALE12, **common)
    if args.domain == "n4_12":  # 4-12 cameras + external conditions; selection at 8 (middle of the range)
        cfg = SimConfig(a_min=args.a_min, lagrange=args.lagrange, **N4_12, **common)
        select = SimConfig(a_min=args.a_min, lagrange=args.lagrange, **N_EVAL[8], **common)
    if args.domain == "testbed":
        cfg = SimConfig(a_min=args.a_min, lagrange=args.lagrange, edges=(2, 4), network_ms=(3, 6), wan_p=0,
                        cloud_background=(.8, 1), **common)
    test = QualityTable.load(args.table, args.scene, split="test")
    if args.cmd == "train":
        # Checkpoint selection on every 5th train sequence, never on the test split.
        # shortcut: not stratified by weather, upgrade if validation F1 drifts far from train F1.
        full = QualityTable.load(args.table, args.scene, split="train")
        names = sorted(full.sequences)
        val = names[::5]
        train(full.subset([s for s in names if s not in val]), args.out, iters=args.iters, cfg=cfg,
              seed=args.seed, eval_table=full.subset(val), tolerance=args.tolerance, lagrange_lr=args.lagrange_lr,
              select_cfg=select, curriculum=args.curriculum, team_mix=args.team_mix)
        print(f"ACTOR -> {args.out}")
    else:
        actors = {f"actor:{Path(a).stem}": load_actor(a) for a in args.actor}
        # real3 = the measured testbed (validated by scripts/sim_replay.py): LAN, whole GPU; randomized = robustness.
        for scenario, cfg in (("real3", SimConfig(a_min=args.a_min, real_edges_only=True, network_ms=(3, 3),
                                                  wan_p=0, cloud_background=(1, 1))),
                              ("randomized", SimConfig(a_min=args.a_min)),
                              ("scale8", SimConfig(a_min=args.a_min, **SCALE8)),
                              ("scale12", SimConfig(a_min=args.a_min, **SCALE12)),
                              ("scale16", SimConfig(a_min=args.a_min, **SCALE16)),
                              ("dyn8", SimConfig(a_min=args.a_min, **DYN8)),
                              ("dyn12", SimConfig(a_min=args.a_min, **DYN12))):
            for name in BASELINES:
                print(json.dumps({"scenario": scenario, **evaluate(test, policy=name, cfg=cfg)}), flush=True)
            for name, actor in actors.items():
                print(json.dumps({"scenario": scenario, **evaluate(test, actor, policy="actor", cfg=cfg),
                                  "policy": name}), flush=True)


if __name__ == "__main__":
    main()
