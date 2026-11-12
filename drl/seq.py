"""Sequential Cloud scheduler (simulator experiment): ONE shared network at the Cloud chooses (model, analytics FPS)
for each camera in turn every 5 s decision; each choice sees the camera's own observation plus how much GPU the
cameras chosen before it in this round already took (Multi-Agent-Transformer-style sequential decisions, so
cameras no longer pile onto the GPU together as in decentralized MAPPO). Team reward: mean TA@200.

  python -B -m drl.seq train --table T.jsonl --scene S.jsonl --out seq.json [--iters 400 --seed 0]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import random

import numpy as np

from drl import central
from drl.sim import CLOUD_MODES, MultiEdgeSim, QualityTable, SimConfig
from edge_agent.drl_policy import observation_v2

K_SCALE = 12  # camera count normaliser


def _cfg(cfg):
    return dataclasses.replace(cfg, async_share=1.0, coord_features=False, priority_feature=False)


def camera_obs(sim, i, committed, chosen_cloud, position):
    """Camera i's live observation + its FPS and network + round context (GPU already committed this round)."""
    e, n = sim.edges[i], len(sim.edges)
    _share, busy = sim.coordination()
    own = observation_v2(e.history, e.mode)
    net = sim.network_ms if e.net_ms is None else e.net_ms
    # own expected Cloud gain per Cloud model (from its encoder, link and Edge speed: what made central work)
    gains = [4 * central.cloud_benefit(sim, e, m) for m in central.MODELS]
    ctx = [e.fps / sim.cfg.target_fps, min(1.0, net / 100), n / K_SCALE, busy, min(2.0, committed) / 2,
           chosen_cloud / n, position / n, float(sim.stale()), *gains, e.edge_factor()]
    return np.concatenate([own, np.array(ctx)]).astype(np.float32)


def run_episode(sim, choose, deadline=200, rng=None):
    """choose(obs, mask) -> action index. Returns (rounds: [[(obs, mask, a)...], reward], parts)."""
    rng = rng or random.Random(0)
    caps = sim.cfg.cloud_capacity
    sim.reset()
    rounds, parts, done = [], [], False
    while not done:
        n = len(sim.edges)
        order = list(range(n))
        rng.shuffle(order)
        acts, steps, committed, on_cloud = [0] * n, [], 0.0, 0
        stale = sim.stale()
        for pos, i in enumerate(order):
            obs = camera_obs(sim, i, committed, on_cloud, pos)
            mask = np.array([not (m in CLOUD_MODES and stale) for m, _f in sim.actions])
            a = choose(obs, mask)
            acts[i] = a
            mode, fps = sim.actions[a]
            if mode in CLOUD_MODES:
                committed += fps / caps[mode]
                on_cloud += 1
            steps.append((obs, mask, a))
        _o, _r, done, info = sim.step(acts)
        reward = float(np.mean([p["ta"][deadline] for p in info["edges"]]))
        rounds.append((steps, reward))
        parts.extend(info["edges"])
    return rounds, parts


def evaluate_seq(table, choose, cfg, *, seed=1234, episodes=12, deadline=200):
    sim = MultiEdgeSim(table, _cfg(cfg), seed=seed)
    rng, parts = random.Random(seed), []
    for _ in range(episodes):
        parts += run_episode(sim, choose, deadline, rng)[1]
    return {f"ta_{deadline}": float(np.mean([p["ta"][deadline] for p in parts])),
            "cloud_share": float(np.mean([p["mode"] in CLOUD_MODES for p in parts])), "parts": parts}


def imitation_data(sim, teacher, episodes, rng, rank="benefit"):
    """(obs, mask, label) per camera decision from the centralized (model, FPS, k) policy, in the sequential
    format: warm start so per-camera PPO begins at the central policy instead of from scratch."""
    caps, acts, data = sim.cfg.cloud_capacity, central.central_actions(sim.cfg.fps_levels), []
    for _ in range(episodes):
        sim.reset()
        current, done = (central.MODELS[0], sim.cfg.target_fps, 0), False
        while not done:
            n = len(sim.edges)
            a = teacher(central.central_obs(sim, current, rank), central.central_mask(acts, n, sim.stale()))
            current = acts[a]
            labels = central.apply(sim, current, rank)
            order = list(range(n))
            rng.shuffle(order)
            committed, on_cloud, stale = 0.0, 0, sim.stale()
            for pos, i in enumerate(order):
                mask = np.array([not (m in CLOUD_MODES and stale) for m, _f in sim.actions])
                data.append((camera_obs(sim, i, committed, on_cloud, pos), mask, labels[i]))
                mode, fps = sim.actions[labels[i]]
                if mode in CLOUD_MODES:
                    committed += fps / caps[mode]
                    on_cloud += 1
            _o, _r, done, _info = sim.step(labels)
    return data


class SeqActor:
    def __init__(self, spec):
        self.w = [(np.array(l["w"]), np.array(l["b"])) for l in spec["layers"]]

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


def train(table, out, *, cfg, iters=400, episodes=8, epochs=6, lr=3e-4, clip=.2, entropy=.01, gamma=.9, seed=0,
          eval_table=None, select_cfgs=(), log=print, teacher=None, bc_episodes=40, bc_epochs=60):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    sim = MultiEdgeSim(table, _cfg(cfg), seed=seed)
    sim.reset()
    obs_dim, n_act = len(camera_obs(sim, 0, 0.0, 0, 0)), len(sim.actions)
    net = lambda out_dim: torch.nn.Sequential(torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),  # noqa: E731
                                              torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, out_dim))
    actor, critic = net(n_act), net(1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    rng, best = random.Random(seed), None

    def sample(obs, mask):
        with torch.no_grad():
            logits = actor(torch.as_tensor(obs)).masked_fill(~torch.as_tensor(mask), -1e9)
            return int(torch.distributions.Categorical(logits=logits).sample())

    def export():
        layers = [m for m in actor if isinstance(m, torch.nn.Linear)]
        return {"format": "seq-v1", "actions": [list(a) for a in sim.actions],
                "layers": [{"w": l.weight.detach().numpy().tolist(), "b": l.bias.detach().numpy().tolist()}
                           for l in layers]}

    def evaluate():
        spec = export()
        return spec, float(np.mean([evaluate_seq(eval_table, SeqActor(spec), c, episodes=3)["ta_200"]
                                    for c in select_cfgs]))

    if teacher is not None:
        data = imitation_data(sim, teacher, bc_episodes, rng)
        O = torch.as_tensor(np.stack([d[0] for d in data]))
        M = torch.as_tensor(np.stack([d[1] for d in data]))
        A = torch.as_tensor([d[2] for d in data])
        bc = torch.optim.Adam(actor.parameters(), lr=1e-3)
        for ep in range(bc_epochs):
            loss = torch.nn.functional.cross_entropy(actor(O).masked_fill(~M, -1e9), A)
            bc.zero_grad()
            loss.backward()
            bc.step()
        acc = float((actor(O).masked_fill(~M, -1e9).argmax(-1) == A).float().mean())
        row = {"imitation_samples": len(data), "bc_loss": round(float(loss), 4), "bc_acc": round(acc, 3)}
        if eval_table is not None:
            spec, best = evaluate()
            Path(out).write_text(json.dumps(spec), encoding="utf-8")
            row["eval_ta200"] = round(best, 4)
        log(json.dumps(row), flush=True)

    for it in range(iters):
        O, M, A, G = [], [], [], []
        rewards = []
        for _ in range(episodes):
            rounds, _parts = run_episode(sim, sample, rng=rng)
            ret = 0.0
            for steps, r in reversed(rounds):
                ret = r + gamma * ret
                for obs, mask, a in steps:
                    O.append(obs); M.append(mask); A.append(a); G.append(ret)
                rewards.append(r)
        O = torch.as_tensor(np.stack(O))
        M = torch.as_tensor(np.stack(M))
        A = torch.as_tensor(A)
        G = torch.as_tensor(G, dtype=torch.float32)
        with torch.no_grad():
            adv = G - critic(O).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            old = torch.distributions.Categorical(logits=actor(O).masked_fill(~M, -1e9)).log_prob(A)
        for _ in range(epochs):
            dist = torch.distributions.Categorical(logits=actor(O).masked_fill(~M, -1e9))
            ratio = torch.exp(dist.log_prob(A) - old)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            vloss = ((critic(O).squeeze(-1) - G) ** 2).mean()
            loss = pg + .5 * vloss - entropy * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        cloud = float(np.mean([sim.actions[int(a)][0] in CLOUD_MODES for a in A]))
        row = {"iter": it, "train_ta200": round(float(np.mean(rewards)), 4), "cloud_share": round(cloud, 3)}
        if eval_table is not None and (it % 10 == 9 or it == iters - 1):
            spec, score = evaluate()
            row["eval_ta200"] = round(score, 4)
            if best is None or score > best:
                best = score
                Path(out).write_text(json.dumps(spec), encoding="utf-8")
        log(json.dumps(row), flush=True)
    if best is None:
        Path(out).write_text(json.dumps(export()), encoding="utf-8")


def main(argv=None):
    from drl import mappo
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train",))
    p.add_argument("--table", nargs="+", required=True)
    p.add_argument("--scene")
    p.add_argument("--out", default="seq.json")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", default="", help="central actor JSON to imitate first (warm start)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=.01)
    args = p.parse_args(argv)
    fl = (5, 10, 15)
    cfg = SimConfig(a_min=.75, **mappo.DIVERSE, fps_levels=fl)
    select = [SimConfig(a_min=.75, **mappo.N_TESTBED[n], fps_levels=fl) for n in (6, 10)] + \
             [SimConfig(a_min=.75, **mappo.N_DIVERSE[8], fps_levels=fl)]
    full = QualityTable.load(args.table, args.scene, split="train")
    names = sorted(full.sequences)
    val = names[::5]
    train(full.subset([s for s in names if s not in val]), args.out, cfg=cfg, iters=args.iters, seed=args.seed,
          episodes=args.episodes, eval_table=full.subset(val), select_cfgs=select, lr=args.lr, entropy=args.entropy,
          teacher=central.CentralActor.load(args.teacher) if args.teacher else None)
    print(f"ACTOR -> {args.out}")


if __name__ == "__main__":
    main()
