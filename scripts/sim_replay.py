"""Simulator-vs-live gate: replay the measured per-second model sequence of explore runs in drl.sim.

  python -B scripts/sim_replay.py WINDOWS.jsonl TABLE.jsonl [--max-error .1]

WINDOWS.jsonl comes from scripts/table_b.py. Each run's Edge is simulated with its real profile (no jitter,
whole Cloud GPU, LAN network, no outages), forced into the model it actually ran each second. Compares
per Edge x mode: timely ratio, p50 latency, encoder load and hottest-core CPU. Gate S1: mean absolute
timely error per Edge x mode <= --max-error.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace  # noqa: E402

from drl.sim import ENC_LOAD, EDGE_FACTOR_ENC_FPS, REAL_EDGES, MultiEdgeSim, QualityTable, SimConfig, _Edge  # noqa: E402


def encode_fps_from_load(load):
    """Invert the simulator's encoder-load model (monotone in encode_fps) by bisection."""
    lo, hi = 1.0, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ENC_LOAD[0] / mid * max(1.0, EDGE_FACTOR_ENC_FPS / mid) ** ENC_LOAD[1] > load:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
from edge_agent.drl_policy import ACTION_MODES  # noqa: E402


def replay(rows, table, seed=0, cloud_inflight=1):
    """-> {(edge, mode): {"live": [...], "sim": [...]}} of per-second dicts (timely, latency, encoder, cpu)."""
    profiles = {p.name: p for p in REAL_EDGES}
    cfg = SimConfig(jitter=0, random_profile_p=0, outage_p=0, cloud_background=(1, 1), network_ms=(3, 3), wan_p=0,
                    cloud_inflight=cloud_inflight)
    runs = defaultdict(list)
    for r in rows:
        runs[r["run"]].append(r)
    # Runs that started together share the Cloud: group by start minute (run name YYYYmmdd_HHMMSS_...).
    groups = defaultdict(list)
    for name, ws in runs.items():
        groups[name[:13]].append(sorted(ws, key=lambda w: w["window"]))
    out = defaultdict(lambda: {"live": [], "sim": []})
    for runs_together in groups.values():
        sim = MultiEdgeSim(table, cfg, seed=seed)
        sim.reset()
        sim.background, sim.network_ms = 1.0, 3.0
        # Each run's encoder capacity from its own measured encoder load (an observed state feature): the
        # same Edge varied a lot between runs (252: load 1.1 in the afternoon, 0.29 in the evening).
        def profile(ws):
            loads = [w["encoder_load"] for w in ws if w["encoder_load"] is not None]
            base = profiles[ws[0]["edge"]]
            return replace(base, encode_fps=encode_fps_from_load(float(np.median(loads)))) if loads else base
        sim.edges = [_Edge(profile(ws), random.Random(seed + k), table, cfg) for k, ws in enumerate(runs_together)]
        for t in range(max(len(ws) for ws in runs_together)):
            live = [ws[t] if t < len(ws) else None for ws in runs_together]
            for e, w in zip(sim.edges, live):
                if w is not None and w["mode"] != e.mode:
                    e.mode, e.since_switch = w["mode"], 0
            sim._tick()
            for e, w in zip(sim.edges, live):
                if w is None:
                    continue
                s = e.last
                key = (w["edge"], w["mode"])
                out[key]["live"].append({"timely": w["timely"], "latency": w["latency_ms"],
                                         "encoder": w["encoder_load"], "cpu": w["cpu_max"]})
                out[key]["sim"].append({"timely": s["timely"] / cfg.target_fps,
                                        "latency": s["latency"] if s["done"] > 0 else None,
                                        "encoder": e.encoder_load(), "cpu": s["cpu"]})
    return out


def summarize(out):
    def agg(xs, key, fn):
        v = [x[key] for x in xs if x[key] is not None]
        return float(fn(v)) if v else float("nan")
    lines = ["| edge | mode | s | timely live / sim | p50 latency ms live / sim | encoder load live / sim | cpu % live / sim |",
             "|---|---|---|---|---|---|---|"]
    errors = {}
    for (edge, mode) in sorted(out, key=lambda k: (k[0], ACTION_MODES.index(k[1]))):
        live, sim = out[(edge, mode)]["live"], out[(edge, mode)]["sim"]
        t_live, t_sim = agg(live, "timely", np.mean), agg(sim, "timely", np.mean)
        errors[(edge, mode)] = abs(t_live - t_sim)
        lines.append(f"| {edge} | {mode} | {len(live)} | {t_live:.3f} / {t_sim:.3f} | "
                     f"{agg(live, 'latency', np.median):.0f} / {agg(sim, 'latency', np.median):.0f} | "
                     f"{agg(live, 'encoder', np.mean):.2f} / {agg(sim, 'encoder', np.mean):.2f} | "
                     f"{agg(live, 'cpu', np.mean):.0f} / {agg(sim, 'cpu', np.mean):.0f} |")
    return lines, errors


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("windows")
    p.add_argument("table")
    p.add_argument("--max-error", type=float, default=.1)
    p.add_argument("--cloud-inflight", type=int, default=1, help="tickets in flight the measured runs used")
    args = p.parse_args(argv)
    rows = [json.loads(x) for x in Path(args.windows).read_text(encoding="utf-8").splitlines()]
    table = QualityTable.load([args.table], split="train")
    lines, errors = summarize(replay(rows, table, cloud_inflight=args.cloud_inflight))
    print("\n".join(lines))
    worst = max(errors.items(), key=lambda kv: kv[1])
    verdict = "PASS" if worst[1] <= args.max_error else "FAIL"
    print(f"[S1] {verdict} worst timely error {worst[1]:.3f} at {worst[0]} (limit {args.max_error})")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
