"""Timely accuracy TA@D on measured runs: mean over the frames the Edge had to analyze of F1 x 1[result <= D ms].

  python -B scripts/timely_accuracy.py RESULTS/tableB_x TABLE.jsonl PLAYLIST.json [--deadlines 100 200 300 500]

Per 1 s window: N = target FPS x window length (frames to analyze, dropped ones included), Q = per-second F1 of
the model that ran (UA-DETRAC table, scene tag of the window), I = completion within D ms of capture
(source_completion_latency_ms). Reported per Edge x mode, plus each Edge's latency percentiles.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrac  # noqa: E402
from table_b import MIN_WINDOWS  # noqa: E402

TABLE_MODEL = {"yolo_edge": "yolov8n", "yolo11s_edge": "yolo11s", "rtdetr_cloud": "rtdetrv2", "lwdetr_large_cloud": "lwdetr",
               "dfine_x_cloud": "dfine_x"}


def windows_with_frames(root, manifest):
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
                break  # Cloud GPU fault: later windows do not describe the system
            yield run.name, w


def timely_accuracy(rows, f1, deadlines):
    """rows: [(edge, mode, scene, n_frames_needed, [latencies of accepted frames])] -> {(edge, mode): {D: TA}}."""
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for edge, mode, scene, needed, lat in rows:
        q = f1.get((scene["sequence"], scene["second"], TABLE_MODEL[mode])) if scene else None
        if q is None or needed <= 0:
            continue
        for d in deadlines:
            cell = acc[(edge, mode)][d]
            cell[0] += q * sum(1 for x in lat if x <= d)
            cell[1] += needed
    return {k: {d: s / n for d, (s, n) in v.items()} for k, v in acc.items()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("table")
    p.add_argument("manifest")
    p.add_argument("--deadlines", type=int, nargs="+", default=[100, 200, 300, 500])
    args = p.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    f1 = {}
    for line in Path(args.table).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        f1[(r["sequence"], r["window"], r["model"])] = r["f1"]
    rows, lat_by = [], defaultdict(list)
    for _run, w in windows_with_frames(args.root, manifest):
        mode, fps = w["context"]["model_mode"], w["context"]["target_fps"]
        lat = [f["source_completion_latency_ms"] for f in w["frames"]
               if f.get("status") == "accepted" and f.get("model_mode") == mode and f.get("source_completion_latency_ms") is not None]
        rows.append((w["stream_key"][0], mode, w.get("scene"), round(fps * w["window_duration_s"]), lat))
        lat_by[(w["stream_key"][0], mode)].extend(lat)
    ta = timely_accuracy(rows, f1, args.deadlines)
    print("| edge | mode | " + " | ".join(f"TA@{d}" for d in args.deadlines) + " | latency p50 / p95 ms |")
    print("|---|---|" + "---|" * len(args.deadlines) + "---|")
    for (edge, mode) in sorted(ta, key=lambda k: (k[0], list(TABLE_MODEL).index(k[1]))):
        lat = lat_by[(edge, mode)]
        print(f"| {edge} | {mode} | " + " | ".join(f"{ta[(edge, mode)][d]:.3f}" for d in args.deadlines)
              + f" | {np.median(lat):.0f} / {np.percentile(lat, 95):.0f} |")


if __name__ == "__main__":
    main()
