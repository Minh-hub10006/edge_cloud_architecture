"""Measured TA@D per multi-stream experiment (runs started together on several Edges = one experiment).

  python -B scripts/n_streams_eval.py RESULTS/tableB_N2 TABLE_FPS.jsonl PLAYLIST.json [--utc-offset edge-252=7]
      [--gap-s 90] [--deadlines 100 200 300]

Runs are grouped by start time (run directory name, shifted by --utc-offset for Edges whose clock is UTC); a new
experiment starts after a gap longer than --gap-s. Per stream TA@D = sum over windows of F1(model, FPS) x frames done
within D ms, divided by frames to analyze; the experiment's TA is the mean over its streams. GPU max utilisation /
power / temperature come from the Cloud gpu.csv over the experiment's span.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timely_accuracy import TABLE_MODEL, windows_with_frames  # noqa: E402

LABEL = {"yolo_edge": "Y8", "yolo11s_edge": "Y11", "rtdetr_cloud": "RT", "lwdetr_large_cloud": "LW", "dfine_x_cloud": "DX"}


def run_start(name, offsets):
    """20261014_225448_edge-246_lwdetr -> (local datetime, edge, policy)."""
    day, clock, edge, policy = name.split("_", 3)
    t = datetime.strptime(day + clock, "%Y%m%d%H%M%S") + timedelta(hours=offsets.get(edge, 0))
    return t, edge, policy


def group(starts, gap_s):
    """{run: start} -> list of run lists, split where consecutive starts are more than gap_s apart."""
    out, last = [], None
    for run, t in sorted(starts.items(), key=lambda kv: kv[1]):
        if last is None or (t - last).total_seconds() > gap_s:
            out.append([])
        out[-1].append(run)
        last = t
    return out


def stream_ta(windows, f1, deadlines):
    """windows: [(mode, fps, scene, needed, latencies)] -> {D: TA} (None if no window has a scene tag)."""
    s, n = defaultdict(float), 0
    for mode, fps, scene, needed, lat, _w in windows:
        q = f1.get((scene["sequence"], scene["second"], TABLE_MODEL[mode], fps)) if scene else None
        if q is None or needed <= 0:
            continue
        n += needed
        for d in deadlines:
            s[d] += q * sum(1 for x in lat if x <= d)
    return {d: s[d] / n for d in deadlines} if n else None


def gpu_span(csv_paths, t0, t1):
    rows = []
    for path in csv_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
            parts = [x.strip() for x in line.split(",")]
            try:
                t = datetime.strptime(parts[0][:19], "%Y/%m/%d %H:%M:%S")
                vals = float(parts[1]), float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue
            if t0 <= t <= t1:
                rows.append(vals)
    if not rows:
        return None
    temp, power, util = np.array(rows).T
    return {"util_mean": float(util.mean()), "util_max": float(util.max()), "power_max": float(power.max()),
            "temp_max": float(temp.max())}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("table")
    p.add_argument("manifest")
    p.add_argument("--utc-offset", nargs="*", default=[], help="EDGE=HOURS for Edges logging in UTC")
    p.add_argument("--gap-s", type=float, default=90)
    p.add_argument("--deadlines", type=int, nargs="+", default=[100, 200, 300])
    p.add_argument("--out", default=None, help="experiments.jsonl (default: ROOT/experiments.jsonl)")
    args = p.parse_args(argv)
    offsets = {k: float(v) for k, v in (x.split("=") for x in args.utc_offset)}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    f1 = {}
    for line in Path(args.table).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        f1[(r["sequence"], r["window"], r["model"], r["fps"])] = r["f1"]
    per_run, info = defaultdict(list), {}
    for run, w in windows_with_frames(args.root, manifest):
        mode, fps = w["context"]["model_mode"], w["context"]["target_fps"]
        lat = [f["source_completion_latency_ms"] for f in w["frames"]
               if f.get("status") == "accepted" and f.get("model_mode") == mode and f.get("source_completion_latency_ms") is not None]
        per_run[run].append((mode, round(fps), w.get("scene"), round(fps * w["window_duration_s"]), lat, w["window_id"]))
    starts = {}
    for run in per_run:
        t, edge, policy = run_start(run, offsets)
        starts[run], info[run] = t, (edge, policy)
    gpu_csv = sorted(Path(args.root).glob("cloud-*/*/gpu.csv"))
    out_path = Path(args.out or Path(args.root) / "experiments.jsonl")
    print("| # | start | N | overlap s | streams (Edge: policy x count) | " + " | ".join(f"TA@{d}" for d in args.deadlines)
          + " | Cloud TA@200 | Edge TA@200 | Cloud lat p50/p95 | GPU util mean/max | W max | °C max |")
    print("|---|---|---|---|---|" + "---|" * len(args.deadlines) + "---|---|---|---|---|---|")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, runs in enumerate(group(starts, args.gap_s), 1):
            tas, cloud, edge_side, cloud_lat, comp, per_stream = [], [], [], [], defaultdict(int), {}
            fps_seen = set()
            # Only the span where every stream runs (runs of different lengths, e.g. 120 s vs 300 s).
            t_in = max(starts[r] for r in runs)
            t_out = min(starts[r] + timedelta(seconds=len(per_run[r])) for r in runs)
            span = {r: [x for x in per_run[r] if t_in <= starts[r] + timedelta(seconds=x[5]) < t_out] for r in runs}
            for run in runs:
                ta = stream_ta(span[run], f1, args.deadlines)
                edge, policy = info[run]
                comp[(edge, policy)] += 1
                fps_seen.update(x[1] for x in span[run])
                if ta is None:
                    continue
                tas.append(ta)
                per_stream[run] = ta[200]
                modes = {x[0] for x in span[run]}
                (cloud if modes <= {"rtdetr_cloud", "lwdetr_large_cloud"} else
                 edge_side if modes <= {"yolo_edge", "yolo11s_edge"} else []).append(ta[200])
                cloud_lat += [v for x in span[run] if x[0].endswith("_cloud") for v in x[4]]
            t0 = min(starts[r] for r in runs)
            gpu = gpu_span(gpu_csv, t0, t0 + timedelta(seconds=200))
            row = {"experiment": i, "start": t0.isoformat(), "n_streams": len(runs),
                   "overlap_s": max(0.0, (t_out - t_in).total_seconds()),
                   "composition": {f"{e}:{p}": c for (e, p), c in sorted(comp.items())},
                   "fps": sorted(fps_seen), "ta": {d: float(np.mean([t[d] for t in tas])) for d in args.deadlines} if tas else None,
                   "ta200_cloud_streams": float(np.mean(cloud)) if cloud else None,
                   "ta200_edge_streams": float(np.mean(edge_side)) if edge_side else None,
                   "cloud_latency_p50": float(np.median(cloud_lat)) if cloud_lat else None,
                   "cloud_latency_p95": float(np.percentile(cloud_lat, 95)) if cloud_lat else None, "gpu": gpu,
                   "runs": runs, "stream_ta200": per_stream}
            f.write(json.dumps(row) + "\n")
            fmt = lambda v, s="{:.3f}": "-" if v is None else s.format(v)  # noqa: E731
            comp_s = ", ".join(f"{e[5:]}:{p}x{c}" for (e, p), c in sorted(comp.items()))
            print(f"| {i} | {t0:%H:%M} | {len(runs)} | {row['overlap_s']:.0f} | {comp_s} | "
                  + " | ".join(fmt(row["ta"][d] if row["ta"] else None) for d in args.deadlines)
                  + f" | {fmt(row['ta200_cloud_streams'])} | {fmt(row['ta200_edge_streams'])}"
                  + f" | {fmt(row['cloud_latency_p50'], '{:.0f}')}/{fmt(row['cloud_latency_p95'], '{:.0f}')}"
                  + (f" | {gpu['util_mean']:.0f}/{gpu['util_max']:.0f} | {gpu['power_max']:.0f} | {gpu['temp_max']:.0f} |"
                     if gpu else " | - | - | - |"))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
