"""Multi-stream contention gate: replay concurrent explore runs on the wall clock through drl.sim.

  python -B scripts/sim_replay_multi.py RESULTS/tableB_multi TABLE.jsonl PLAYLIST.json --since 20261013_2030
      [--utc-offset edge-252=7]

Each run is one simulated Edge (its real profile, encoder capacity inferred from its measured encoder load);
runs are aligned by start time (run directory name; hosts whose clock is UTC get --utc-offset hours).
Every wall-clock second the simulator ticks with the models the live streams actually ran. Compares Cloud-mode
timely ratio and capture-to-result latency by the number of streams on the Cloud at that second.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drl.sim import CLOUD_MODES, REAL_EDGES, MultiEdgeSim, QualityTable, SimConfig, _Edge  # noqa: E402
from sim_replay import encode_fps_from_load  # noqa: E402
from timely_accuracy import windows_with_frames  # noqa: E402


def load_runs(root, manifest, since, utc_offset):
    runs = defaultdict(list)
    for name, w in windows_with_frames(root, manifest):
        runs[name].append(w)
    out = []
    for name, ws in runs.items():
        edge = ws[0]["stream_key"][0]
        start = datetime.strptime(name[:15], "%Y%m%d_%H%M%S") + timedelta(hours=utc_offset.get(edge, 0))
        if start < since:
            continue
        out.append({"name": name, "edge": edge, "start": start, "windows": ws})
    return out


def replay(runs, table, cloud_inflight=2, seed=0):
    profiles = {p.name: p for p in REAL_EDGES}
    cfg = SimConfig(jitter=0, random_profile_p=0, outage_p=0, cloud_background=(1, 1), network_ms=(3, 3), wan_p=0,
                    cloud_inflight=cloud_inflight)
    sim = MultiEdgeSim(table, cfg, seed=seed)
    sim.reset()
    sim.background, sim.network_ms = 1.0, 3.0
    edges = []
    for k, r in enumerate(runs):
        loads = [w["raw"]["encoder_write_ms"] * 15 / 1000 for w in r["windows"] if w["raw"].get("encoder_write_ms")]
        p = replace(profiles[r["edge"]], encode_fps=encode_fps_from_load(float(np.median(loads))))
        edges.append(_Edge(p, random.Random(seed + k), table, cfg))
    sim.edges = edges
    t0 = min(r["start"] for r in runs)
    t1 = max(r["start"] + timedelta(seconds=len(r["windows"])) for r in runs)
    out = []
    for t in range(int((t1 - t0).total_seconds())):
        now = t0 + timedelta(seconds=t)
        live = []
        for r, e in zip(runs, edges):
            k = int((now - r["start"]).total_seconds())
            w = r["windows"][k] if 0 <= k < len(r["windows"]) else None
            mode = w["context"]["model_mode"] if w else "yolo_edge"  # idle stream: off the Cloud
            if mode != e.mode:
                e.mode, e.since_switch = mode, 0
            live.append(w)
        on_cloud = sum(1 for w in live if w and w["context"]["model_mode"] in CLOUD_MODES)
        sim._tick()
        for r, e, w in zip(runs, edges, live):
            if w is None or w["context"]["model_mode"] not in CLOUD_MODES:
                continue
            mode = w["context"]["model_mode"]
            lat = [f["source_completion_latency_ms"] for f in w["frames"]
                   if f.get("status") == "accepted" and f.get("model_mode") == mode and f.get("source_completion_latency_ms")]
            out.append({"edge": r["edge"], "mode": mode, "cloud_streams": on_cloud,
                        "live_timely": w["raw"]["timely_fps"] / w["context"]["target_fps"],
                        "sim_timely": e.last["timely"] / cfg.target_fps,
                        "live_latency": float(np.median(lat)) if lat else None, "sim_latency": e.last["completion"],
                        "live_pressure": w["raw"].get("cloud_arrival_pressure_raw"), "sim_pressure": sim.cloud_pressure})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("table")
    p.add_argument("manifest")
    p.add_argument("--since", required=True, help="YYYYmmdd_HHMM, local time")
    p.add_argument("--utc-offset", nargs="*", default=[], help="edge=hours for hosts whose run names are UTC")
    args = p.parse_args(argv)
    offsets = {k: float(v) for k, v in (x.split("=") for x in args.utc_offset)}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    runs = load_runs(args.root, manifest, datetime.strptime(args.since, "%Y%m%d_%H%M"), offsets)
    for r in sorted(runs, key=lambda r: r["start"]):
        print(f"run {r['name']} edge={r['edge']} start={r['start']:%H:%M:%S} seconds={len(r['windows'])}")
    rows = replay(runs, QualityTable.load([args.table], split="train"))
    print("\n| streams on Cloud | seconds | timely live / sim | latency p50 ms live / sim | Cloud pressure live / sim |")
    print("|---|---|---|---|---|")
    for n in sorted({r["cloud_streams"] for r in rows}):
        g = [r for r in rows if r["cloud_streams"] == n]
        med = lambda k: np.median([r[k] for r in g if r[k] is not None])  # noqa: E731
        print(f"| {n} | {len(g)} | {np.mean([r['live_timely'] for r in g]):.3f} / {np.mean([r['sim_timely'] for r in g]):.3f} | "
              f"{med('live_latency'):.0f} / {med('sim_latency'):.0f} | {med('live_pressure'):.2f} / {med('sim_pressure'):.2f} |")
    print("\n| edge | mode | seconds | timely live / sim | latency p50 ms live / sim |\n|---|---|---|---|---|")
    for key in sorted({(r["edge"], r["mode"]) for r in rows}):
        g = [r for r in rows if (r["edge"], r["mode"]) == key]
        med = lambda k: np.median([r[k] for r in g if r[k] is not None])  # noqa: E731
        print(f"| {key[0]} | {key[1]} | {len(g)} | {np.mean([r['live_timely'] for r in g]):.3f} / "
              f"{np.mean([r['sim_timely'] for r in g]):.3f} | {med('live_latency'):.0f} / {med('sim_latency'):.0f} |")


if __name__ == "__main__":
    main()
