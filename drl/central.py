"""Centralized Cloud scheduler (simulator experiment): ONE agent at the Cloud picks, every 5 s decision,
(Cloud model, analytics FPS, k = how many cameras go to the Cloud); the k cameras with the strongest encoders run that
pair, the others YOLO11s at 15 FPS on their Edge. Motivation: decentralized MAPPO (v1-v13) collapses to Edge-only
because Edges exploring the Cloud at the same time saturate the GPU (relative overgeneralization); a single agent
that sees the GPU state assigns the roles directly.

  python -B -m drl.central train --table T.jsonl --scene S.jsonl --out central.pt.json [--iters 300 --seed 0]

Observation (what the Cloud can measure): cameras, GPU busy share incl. other tenants, own arrival pressure,
network delay, stale flag, last-window timely ratio on Cloud / on Edge, share of weak encoders, current action.
Reward: mean TA@200 over the cameras (the evaluation metric).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drl.sim import CLOUD_MODES, EDGE_CAP_KEY, PRE_WORKER_MS, MultiEdgeSim, QualityTable, SimConfig, on_time_share

MODELS = ("rtdetr_cloud", "lwdetr_large_cloud")
LOCAL = ("yolo11s_edge", 15)
K_MAX = 12
EDGE_FACTOR_ENC = 25.5  # encoder FPS below which an Edge cannot publish 15 FPS while analyzing (sim.EDGE_FACTOR_ENC_FPS)
# Expected F1 gain of a Cloud slot (LW-DETR@15 minus YOLO11s@15) from label-free features the Edge observes,
# least squares on the TRAIN split: [1, brightness/255, motion/50, vehicles/50, box_change]; test corr .39, top-40%
# by prediction gains .115 F1 per slot vs .072 random (oracle .166). Night/rain cameras gain most.
GAIN_COEF = (.138, -.253, -.431, .069, .533)


def predicted_gain(e):
    raw = e.history[-1]["raw"] if e.history else {}
    x = (1.0, (raw.get("brightness") or 128) / 255, (raw.get("motion") or 0) / 50, (raw.get("vehicle_count") or 0) / 50,
         raw.get("box_change_mean") or 0)
    # weak encoders lose Cloud frames they never publish (sim: Cloud rate x edge_factor ** .5)
    return sum(c * v for c, v in zip(GAIN_COEF, x)) * e.edge_factor() ** .5


def central_actions(fps_levels):
    """(model, fps, k) per action index; k = 0 means every camera local."""
    return [(m, f, k) for k in range(K_MAX + 1) for m in MODELS for f in fps_levels]


def central_mask(actions, n, stale):
    return np.array([k <= n and (k == 0 or not stale) for _m, _f, k in actions])


# Mean F1 at 15 FPS on the TRAIN split (results/detrac/table_fps.jsonl): what a camera expects from each model.
MEAN_F1 = {"lwdetr_large_cloud": .876, "rtdetr_cloud": .816, "yolo11s_edge": .791}


def cloud_benefit(sim, e, model="lwdetr_large_cloud", deadline=200):
    """Expected TA gain of putting camera e on the Cloud instead of YOLO11s on its Edge, from what the Cloud
    measures per camera: its Cloud round trip without GPU queueing (publish/join delay of its encoder + its own
    link) and its Edge inference speed (slows under CPU contention / throttling)."""
    net = sim.network_ms if e.net_ms is None else e.net_ms
    pre = PRE_WORKER_MS[0] + PRE_WORKER_MS[1] * 1000 / e.enc_fps()
    cloud_ms = e.join_ms() + 1000 / sim.cfg.cloud_capacity[model] + max(0.0, net - 3.0) + pre
    cap = getattr(e.p, EDGE_CAP_KEY["yolo11s_edge"]) * e.slow() * e.cap_mult
    local_ms = (1000 / cap + 6) / e.edge_factor() + pre
    local_rate = min(1.0, cap / e.fps, e.edge_factor())
    return (MEAN_F1[model] * e.edge_factor() ** .5 * on_time_share(cloud_ms, deadline)
            - MEAN_F1["yolo11s_edge"] * local_rate * on_time_share(local_ms, deadline))


def central_obs(sim, current, rank="encoder"):
    n = len(sim.edges)
    _share, busy = sim.coordination()
    def timely(on_cloud):
        xs = [e.history[-1]["raw"]["timely_fps"] / e.history[-1]["context"]["target_fps"]
              for e in sim.edges if (e.mode in CLOUD_MODES) == on_cloud and e.history]
        return (float(np.mean(xs)), 1.0) if xs else (0.0, 0.0)
    tc, has_c = timely(True)
    te, has_e = timely(False)
    weak = float(np.mean([e.enc_fps() < EDGE_FACTOR_ENC for e in sim.edges]))
    model, fps, k = current
    extra = []
    if rank == "gain":  # how much the cameras would gain on the Cloud right now
        g = sorted((predicted_gain(e) for e in sim.edges), reverse=True)
        extra = [float(np.mean(g)) * 5, float(np.mean(g[:max(1, n // 4)])) * 5]
    elif rank == "benefit":  # per-camera Cloud benefit profile: best, quartiles, share of cameras that gain
        b = sorted((cloud_benefit(sim, e) for e in sim.edges), reverse=True)
        q = lambda f: b[min(n - 1, int(f * n))]  # noqa: E731
        extra = [b[0] * 4, q(.25) * 4, q(.5) * 4, q(.75) * 4, float(np.mean([x > 0 for x in b]))]
    return np.array([*extra, n / K_MAX, busy, min(2.0, sim.cloud_pressure) / 2, min(1.0, sim.network_ms / 60),
                     float(sim.stale()), tc, has_c, te, has_e, weak, k / K_MAX, float(model == MODELS[1]),
                     fps / sim.cfg.target_fps], dtype=np.float32)


def apply(sim, action, rank="encoder"):
    """-> per-Edge action indices: k cameras run (model, fps) on the Cloud, the rest YOLO11s 15 FPS. rank picks
    the k: "encoder" = strongest encoders, "gain" = highest predicted Cloud gain (scene difficulty)."""
    model, fps, k = action
    key = {"encoder": lambda i: -sim.edges[i].enc_fps(), "gain": lambda i: -predicted_gain(sim.edges[i]),
           "benefit": lambda i: -cloud_benefit(sim, sim.edges[i], model)}[rank]
    order = sorted(range(len(sim.edges)), key=key)
    cloud = set(order[:k]) if not sim.stale() else set()
    return [sim.action_index(model, fps) if i in cloud else sim.action_index(*LOCAL) for i in range(len(sim.edges))]


def run_episode(sim, choose, deadline=200, rank="encoder"):
    """choose(obs, mask) -> action index. Returns (per-decision (obs, mask, action, reward), per-camera parts)."""
    acts = central_actions(sim.cfg.fps_levels)
    sim.reset()
    current, steps, parts, done = (MODELS[0], sim.cfg.target_fps, 0), [], [], False
    while not done:
        obs = central_obs(sim, current, rank)
        mask = central_mask(acts, len(sim.edges), sim.stale())
        a = choose(obs, mask)
        current = acts[a]
        _o, _r, done, info = sim.step(apply(sim, current, rank))
        reward = float(np.mean([p["ta"][deadline] for p in info["edges"]]))
        steps.append((obs, mask, a, reward))
        parts.extend(info["edges"])
    return steps, parts


def _central_cfg(cfg):
    # every camera may be reassigned at each decision: the agent owns the roles
    import dataclasses
    return dataclasses.replace(cfg, async_share=1.0, coord_features=False)


def evaluate_central(table, choose, cfg, *, seed=1234, episodes=12, deadline=200, rank=None):
    sim = MultiEdgeSim(table, _central_cfg(cfg), seed=seed)
    rank = rank or getattr(choose, "rank", "encoder")
    parts = []
    for _ in range(episodes):
        parts += run_episode(sim, choose, deadline, rank)[1]
    return {f"ta_{deadline}": float(np.mean([p["ta"][deadline] for p in parts])),
            "cloud_share": float(np.mean([p["mode"] in CLOUD_MODES for p in parts])), "parts": parts}


class CentralActor:
    """numpy forward pass of the trained MLP (JSON weights)."""

    def __init__(self, spec):
        self.w = [(np.array(l["w"]), np.array(l["b"])) for l in spec["layers"]]
        self.rank = spec.get("rank", "encoder")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __call__(self, obs, mask):
        x = obs
        for i, (w, b) in enumerate(self.w):
            x = x @ w.T + b
            if i < len(self.w) - 1:
                x = np.tanh(x)
        return int(np.argmax(np.where(mask, x, -1e9)))


def train(table, out, *, cfg, iters=300, episodes=8, epochs=6, lr=3e-4, clip=.2, entropy=.01, gamma=.95, lam=.9,
          seed=0, eval_table=None, select_cfg=None, log=print, rank="encoder"):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    acts = central_actions(cfg.fps_levels)
    sim = MultiEdgeSim(table, _central_cfg(cfg), seed=seed)
    sim.reset()
    obs_dim = len(central_obs(sim, (MODELS[0], 15, 0), rank))
    actor = torch.nn.Sequential(torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                torch.nn.Tanh(), torch.nn.Linear(64, len(acts)))
    critic = torch.nn.Sequential(torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                 torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    best = None

    def sample(obs, mask):
        with torch.no_grad():
            logits = actor(torch.as_tensor(obs)).masked_fill(~torch.as_tensor(mask), -1e9)
            return int(torch.distributions.Categorical(logits=logits).sample())

    def export():
        layers = [m for m in actor if isinstance(m, torch.nn.Linear)]
        return {"format": "central-v1", "rank": rank, "actions": [list(a) for a in acts],
                "layers": [{"w": l.weight.detach().numpy().tolist(), "b": l.bias.detach().numpy().tolist()}
                           for l in layers]}

    for it in range(iters):
        batch = [run_episode(sim, sample, rank=rank)[0] for _ in range(episodes)]
        O, M, A, R, ADV = [], [], [], [], []
        for steps in batch:
            obs = torch.as_tensor(np.stack([s[0] for s in steps]))
            with torch.no_grad():
                v = critic(obs).squeeze(-1).numpy()
            rew = np.array([s[3] for s in steps])
            adv, last = np.zeros_like(rew), 0.0
            for t in reversed(range(len(steps))):
                nv = v[t + 1] if t + 1 < len(steps) else 0.0
                last = rew[t] + gamma * nv - v[t] + gamma * lam * last
                adv[t] = last
            O.append(obs); M += [s[1] for s in steps]; A += [s[2] for s in steps]; R.append(adv + v); ADV.append(adv)
        O = torch.cat(O)
        M = torch.as_tensor(np.stack(M))
        A = torch.as_tensor(A)
        RET = torch.as_tensor(np.concatenate(R), dtype=torch.float32)
        ADVt = torch.as_tensor(np.concatenate(ADV), dtype=torch.float32)
        ADVt = (ADVt - ADVt.mean()) / (ADVt.std() + 1e-8)
        with torch.no_grad():
            old = torch.distributions.Categorical(logits=actor(O).masked_fill(~M, -1e9)).log_prob(A)
        for _ in range(epochs):
            dist = torch.distributions.Categorical(logits=actor(O).masked_fill(~M, -1e9))
            ratio = torch.exp(dist.log_prob(A) - old)
            pg = -torch.min(ratio * ADVt, ratio.clamp(1 - clip, 1 + clip) * ADVt).mean()
            vloss = ((critic(O).squeeze(-1) - RET) ** 2).mean()
            loss = pg + .5 * vloss - entropy * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        mean_ta = float(np.mean([s[3] for steps in batch for s in steps]))
        k_mean = float(np.mean([acts[s[2]][2] for steps in batch for s in steps]))
        row = {"iter": it, "train_ta200": round(mean_ta, 4), "k_mean": round(k_mean, 2)}
        if eval_table is not None and (it % 10 == 9 or it == iters - 1):
            spec = export()
            # selection on several scenarios (e.g. testbed and external at 6 and 10 cameras): mean TA@200
            sel = select_cfg if isinstance(select_cfg, (list, tuple)) else [select_cfg or cfg]
            score = float(np.mean([evaluate_central(eval_table, CentralActor(spec), c, episodes=3)["ta_200"]
                                   for c in sel]))
            row["eval_ta200"] = round(score, 4)
            if best is None or score > best:
                best = score
                Path(out).write_text(json.dumps(spec), encoding="utf-8")
        log(json.dumps(row), flush=True) if log is print else log(json.dumps(row))
    if best is None:
        Path(out).write_text(json.dumps(export()), encoding="utf-8")


def main(argv=None):
    from drl import mappo
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train",))
    p.add_argument("--table", nargs="+", required=True)
    p.add_argument("--scene")
    p.add_argument("--out", default="central.json")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps-levels", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--rank", choices=("encoder", "gain", "benefit"), default="encoder",
                   help="which cameras get the k Cloud slots")
    p.add_argument("--domain", choices=("external", "diverse"), default="external")
    args = p.parse_args(argv)
    cfg = SimConfig(a_min=.75, **(mappo.DIVERSE if args.domain == "diverse" else mappo.N4_12),
                    fps_levels=tuple(args.fps_levels))
    fl = tuple(args.fps_levels)
    if args.domain == "diverse":
        select = [SimConfig(a_min=.75, **mappo.N_TESTBED[n], fps_levels=fl) for n in (6, 10)] +                  [SimConfig(a_min=.75, **mappo.N_DIVERSE[n], fps_levels=fl) for n in (6, 10)]
    else:
        select = [SimConfig(a_min=.75, **scen[n], fps_levels=fl) for scen in (mappo.N_TESTBED, mappo.N_EVAL)
                  for n in (6, 10)]
    full = QualityTable.load(args.table, args.scene, split="train")
    names = sorted(full.sequences)
    val = names[::5]
    train(full.subset([s for s in names if s not in val]), args.out, cfg=cfg, iters=args.iters, seed=args.seed,
          episodes=args.episodes,
          eval_table=full.subset(val), select_cfg=select, rank=args.rank)
    print(f"ACTOR -> {args.out}")


if __name__ == "__main__":
    main()
