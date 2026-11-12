"""Table B summary: measured control-v1 explore runs per Edge x model (simulator calibration input).

  python -B scripts/table_b.py RESULTS/tableB PLAYLIST.json [--out RESULTS/tableB]

Reads every PASS run, plus REVIEW runs with >= MIN_WINDOWS windows (a run cut short, e.g. by a Cloud GPU
fault), each up to the first window with a Cloud inference_error. Tags each 1 s window with the
UA-DETRAC scene it showed (detrac.label_windows, honoring <run>/start_frame), and writes
windows.jsonl (one row per window) and TABLE_B.md (per Edge x mode, and Cloud modes by Cloud pressure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrac  # noqa: E402

MODES = ("yolo_edge", "yolo11s_edge", "rtdetr_cloud", "lwdetr_large_cloud")
PRESSURE_BINS = ((0, .15), (.15, .35), (.35, .6), (.6, 10))
MIN_WINDOWS = 300


def windows(root, manifest):
    for verdict in sorted(Path(root).glob("*/*/verdict.json")):
        run = verdict.parent
        if not (run / "state.jsonl").is_file():
            continue  # failed at startup (e.g. model load), no windows
        state = (run / "state.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(verdict.read_text(encoding="utf-8")).get("status") != "PASS" and len(state) < MIN_WINDOWS:
            continue
        marker = run / "start_frame"
        start = int(marker.read_text()) if marker.is_file() else 0
        for w in detrac.label_windows(state, manifest, start):
            if any(f.get("status") == "inference_error" for f in w.get("frames", ())):
                break  # Cloud model failure (GPU fault): later windows do not describe the system
            raw, ctx = w["raw"], w["context"]
            results = (w.get("action_results") or {}).get(ctx["decision_id"]) or {}
            media_fps = w.get("media_fps") or ctx["target_fps"]
            encoder = raw.get("encoder_write_ms")
            yield {"run": run.name, "edge": w["stream_key"][0], "window": w["window_id"], "mode": ctx["model_mode"],
                   "timely": raw["timely_fps"] / ctx["target_fps"], "effective": raw["effective_fps"] / ctx["target_fps"],
                   "latency_ms": raw.get("active_path_latency_ms"), "cpu_max": raw.get("cpu_max_core_pct"),
                   "cpu_total": raw.get("cpu_total_pct"), "temp_c": raw.get("temp_c"),
                   "encoder_load": None if encoder is None else encoder * media_fps / 1000,
                   "edge_inference_ms": raw.get("edge_inference_ms"), "cloud_ms": raw.get("cloud_processing_ms"),
                   "transport_ms": raw.get("transport_overhead_ms"), "backlog": raw.get("edge_backlog_norm"),
                   "pressure": raw.get("cloud_arrival_pressure_raw"), "inflight": raw.get("cloud_inflight_norm"),
                   "failed": results.get("failed", 0), "accepted": results.get("accepted", 0),
                   "scene": w.get("scene")}


def stat(rows, key, fn=np.mean):
    xs = [r[key] for r in rows if r[key] is not None]
    return float(fn(xs)) if xs else float("nan")


def p95(xs):
    return np.percentile(xs, 95)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("manifest")
    p.add_argument("--out")
    args = p.parse_args(argv)
    out = Path(args.out or args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = list(windows(args.root, manifest))
    if not rows:
        raise SystemExit("[B0] FAIL no PASS runs found")
    with open(out / "windows.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    edges = sorted({r["edge"] for r in rows})
    lines = ["# Table B: measured explore runs", "",
             f"{len(rows)} windows (1 s) from {len({r['run'] for r in rows})} runs; "
             f"{sum(r['scene'] is not None for r in rows)} tagged with a UA-DETRAC scene.", "",
             "## Per Edge x mode", "",
             "| edge | mode | s | timely | p50 lat ms | p95 lat ms | cpu max % | temp C | encoder load | edge infer ms | fail/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in edges:
        for m in MODES:
            g = [r for r in rows if r["edge"] == e and r["mode"] == m]
            if g:
                lines.append(f"| {e} | {m} | {len(g)} | {stat(g, 'timely'):.3f} | {stat(g, 'latency_ms', np.median):.0f} | "
                             f"{stat(g, 'latency_ms', p95):.0f} | {stat(g, 'cpu_max'):.0f} | {stat(g, 'temp_c', np.max):.1f} | "
                             f"{stat(g, 'encoder_load'):.2f} | {stat(g, 'edge_inference_ms'):.1f} | {stat(g, 'failed'):.2f} |")
    lines += ["", "## Cloud modes by Cloud arrival pressure (timely / p95 latency ms / n)", "",
              "| edge | mode | " + " | ".join(f"{lo}-{hi if hi < 10 else ''}" for lo, hi in PRESSURE_BINS) + " |",
              "|---|---|" + "---|" * len(PRESSURE_BINS)]
    for e in edges:
        for m in MODES[2:]:
            cells = []
            for lo, hi in PRESSURE_BINS:
                g = [r for r in rows if r["edge"] == e and r["mode"] == m and r["pressure"] is not None and lo <= r["pressure"] < hi]
                cells.append(f"{stat(g, 'timely'):.2f} / {stat(g, 'latency_ms', p95):.0f} / {len(g)}" if g else "-")
            lines.append(f"| {e} | {m} | " + " | ".join(cells) + " |")
    (out / "TABLE_B.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
