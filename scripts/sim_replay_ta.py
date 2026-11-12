"""Per-experiment TA@D, live vs simulator: replay each multi-stream experiment second by second through drl.sim.

  python -B scripts/sim_replay_ta.py RESULTS/tableB_N2 TABLE_FPS.jsonl PLAYLIST.json [--utc-offset edge-252=7]
      [--skip 13] [--deadline 200]

Experiments are grouped as in n_streams_eval.py and scored only over the span where all their streams run. Each
stream is one simulated Edge (its host's profile, encoder capacity inferred from its measured encoder load), running
the model and FPS the live stream ran that second. Both sides use the same per-second F1 (scene the live window
showed), so the gap is the simulator's delivery and latency error only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drl.sim import REAL_EDGES, MultiEdgeSim, QualityTable, SimConfig, _Edge, on_time_share  # noqa: E402
from n_streams_eval import group, run_start  # noqa: E402
from sim_replay import encode_fps_from_load  # noqa: E402
from timely_accuracy import TABLE_MODEL, windows_with_frames  # noqa: E402


def replay_experiment(runs, table, f1, deadline, seed=0):
    """runs: [(start, edge, windows)] -> (live TA, sim TA) means over streams, overlap span only."""
    profiles = {p.name: p for p in REAL_EDGES}
    cfg = SimConfig(jitter=0, random_profile_p=0, outage_p=0, cloud_background=(1, 1), network_ms=(3, 3), wan_p=0)
    sim = MultiEdgeSim(table, cfg, seed=seed)
    sim.reset()
    sim.background, sim.network_ms = 1.0, 3.0
    edges = []
    for k, (_t, edge, ws) in enumerate(runs):
        loads = [w["raw"]["encoder_write_ms"] * 15 / 1000 for w in ws if w["raw"].get("encoder_write_ms")]
        p = replace(profiles[edge], encode_fps=encode_fps_from_load(float(np.median(loads))))
        edges.append(_Edge(p, random.Random(seed + k), table, cfg))
    sim.edges = edges
    t0 = min(t for t, _e, _w in runs)
    t_in = max(t for t, _e, _w in runs)
    t_out = min(t + timedelta(seconds=len(ws)) for t, _e, ws in runs)
    acc = [[0.0, 0.0, 0] for _ in runs]  # live sum, sim sum, frames to analyze
    for s in range(int((t_out - t0).total_seconds())):
        now = t0 + timedelta(seconds=s)
        current = []
        for (t, _edge, ws), e in zip(runs, edges):
            k = int((now - t).total_seconds())
            w = ws[k] if 0 <= k < len(ws) else None
            mode, fps = (w["context"]["model_mode"], round(w["context"]["target_fps"])) if w else ("yolo_edge", 15)
            if mode != e.mode:
                e.mode, e.since_switch = mode, 0
            e.fps = fps
            current.append(w)
        sim._tick()
        if now < t_in:
            continue
        for i, (w, e) in enumerate(zip(current, edges)):
            if w is None or not w.get("scene"):
                continue
            mode, fps = e.mode, e.fps
            q = f1.get((w["scene"]["sequence"], w["scene"]["second"], TABLE_MODEL[mode], fps))
            if q is None:
                continue
            lat = [f["source_completion_latency_ms"] for f in w["frames"] if f.get("status") == "accepted"
                   and f.get("model_mode") == mode and f.get("source_completion_latency_ms") is not None]
            x = e.last
            acc[i][0] += q * sum(1 for v in lat if v <= deadline)
            acc[i][1] += q * x["done"] * on_time_share(x["completion"], deadline)
            acc[i][2] += fps
    live = [a / n for a, _b, n in acc if n]
    sim_ta = [b / n for _a, b, n in acc if n]
    return float(np.mean(live)), float(np.mean(sim_ta))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("table")
    p.add_argument("manifest")
    p.add_argument("--utc-offset", nargs="*", default=[])
    p.add_argument("--gap-s", type=float, default=90)
    p.add_argument("--skip", type=int, nargs="*", default=[], help="experiment numbers to leave out (aborted)")
    p.add_argument("--deadline", type=int, default=200)
    args = p.parse_args(argv)
    offsets = {k: float(v) for k, v in (x.split("=") for x in args.utc_offset)}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    f1 = {}
    for line in Path(args.table).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        f1[(r["sequence"], r["window"], r["model"], r["fps"])] = r["f1"]
    table = QualityTable.load([args.table], split="train")
    per_run = defaultdict(list)
    for run, w in windows_with_frames(args.root, manifest):
        per_run[run].append(w)
    starts = {run: run_start(run, offsets)[0] for run in per_run}
    print(f"| # | N | streams | live TA@{args.deadline} | sim TA@{args.deadline} | sim - live |\n|---|---|---|---|---|---|")
    errors = []
    for i, runs in enumerate(group(starts, args.gap_s), 1):
        if i in args.skip:
            continue
        comp = defaultdict(int)
        for r in runs:
            _t, edge, policy = run_start(r, offsets)
            comp[f"{edge[5:]}:{policy}"] += 1
        live, sim = replay_experiment([(starts[r], run_start(r, offsets)[1], per_run[r]) for r in runs], table, f1,
                                      args.deadline)
        errors.append(sim - live)
        print(f"| {i} | {len(runs)} | {', '.join(f'{k}x{v}' for k, v in sorted(comp.items()))} | {live:.3f} | "
              f"{sim:.3f} | {sim - live:+.3f} |", flush=True)
    print(f"\nmean |sim - live| = {np.mean(np.abs(errors)):.3f}, worst = {max(errors, key=abs):+.3f}")


if __name__ == "__main__":
    main()
