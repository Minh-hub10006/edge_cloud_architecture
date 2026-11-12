"""Aggregate repeated benchmark runs into one comparable table per case.

A case is (model_key, profile, transport, max_inflight). Repetitions are
combined as mean/std; stages are ranked by mean p50 so the dominant cost is
first. Resources are filtered to each run's own measurement interval.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

MAX_CV = 0.05  # B3: repetitions must agree within 5% effective FPS

# Serial lanes: stages that run one frame at a time, so each lane caps FPS at
# 1000 / (sum of its p50 ms). The lane with the lowest cap is the bottleneck.
# shortcut: cloud_gpu assumes one GPU slot; CLOUD_MAX_INFLIGHT=2 can overlap
# two calls, so its true cap may be up to 2x higher.
LANES = {
    "model_call": ("adapter_call_ms",),
    "edge_producer": ("source_next_ms", "edge_preprocess_ms", "edge_tensor_copy_ms"),
    "edge_infer": ("edge_session_call_ms", "edge_postprocess_ms", "edge_normalize_ms"),
    "edge_media": ("ffmpeg_write_ms",),  # resize + identity band + pipe to x264, serial per frame
    "cloud_gpu": ("cloud_preprocess_ms", "cloud_session_call_ms", "cloud_postprocess_ms"),
    "cloud_tracker": ("cloud_bytetrack_update_ms",),
}


def lanes(stages: list[dict[str, Any]], measured_fps: float | None) -> list[dict[str, Any]]:
    p50 = {row["stage"]: row["p50_ms"] for row in stages}
    rows = []
    for lane, members in LANES.items():
        present = [m for m in members if m in p50]
        if not present:
            continue
        busy = sum(p50[m] for m in present)
        cap = 1000 / busy if busy > 0 else None
        rows.append({"lane": lane, "stages": present, "busy_p50_ms": busy, "fps_cap": cap,
                     "load": measured_fps / cap if cap and measured_fps else None})
    rows.sort(key=lambda row: row["fps_cap"] or float("inf"))
    return rows


def _rows(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _window(rows, start, end):
    if start is None or end is None:
        return []
    return [row for row in rows if start <= row.get("mono_ns", -1) <= end]


def _resources(rows) -> dict[str, Any]:
    def values(get):
        return [v for v in (get(r) for r in rows) if isinstance(v, (int, float))]
    cpu = values(lambda r: r.get("host_cpu_percent"))
    rss = values(lambda r: r.get("process_rss_bytes"))
    gpu = values(lambda r: (r.get("gpu") or {}).get("utilization_pct"))
    vram = values(lambda r: (r.get("gpu") or {}).get("memory_used_mib"))
    # Hottest sensor per sample (millidegrees C): Edge thermal headroom for sustained runs.
    temp = values(lambda r: max((r.get("thermal_raw") or {}).values(), default=None))
    return {"samples": len(rows),
            "host_cpu_mean_pct": statistics.fmean(cpu) if cpu else None,
            "process_rss_max_mib": max(rss) / 2**20 if rss else None,
            "gpu_util_mean_pct": statistics.fmean(gpu) if gpu else None,
            "gpu_mem_max_mib": max(vram) if vram else None,
            "temp_max_c": max(temp) / 1000 if temp else None}


def load_runs(roots: list[Path], cloud_resources: Path | None = None) -> list[dict[str, Any]]:
    cloud_rows = _rows(cloud_resources) if cloud_resources else []
    runs = []
    for root in roots:
        for summary_path in sorted(root.rglob("summary.json")):
            directory = summary_path.parent
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            interval = summary.get("measurement_interval") or {}
            local = _rows(directory / "resources.jsonl") or _rows(directory / "resources-edge.jsonl")
            local = _window(local, interval.get("started_mono_ns"), interval.get("finished_mono_ns"))
            tracked = summary.get("bytetrack_completion_interval") or {}
            cloud = _window(cloud_rows, tracked.get("first_mono_ns"), tracked.get("last_mono_ns"))
            runs.append({"dir": str(directory), "summary": summary,
                         "max_inflight": (manifest.get("options") or {}).get("max_inflight"),
                         "source_loop": (manifest.get("identity") or {}).get("source_loop"),
                         "source_prefetch": (manifest.get("identity") or {}).get("source_prefetch") or 0,
                         "resources_local": _resources(local),
                         "resources_cloud": _resources(cloud) if cloud_rows else None})
    return runs


def aggregate(runs: list[dict[str, Any]], expected_reps: int = 3) -> list[dict[str, Any]]:
    cases: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        s = run["summary"]
        scope = s.get("profile") + (f"+prefetch{run['source_prefetch']}" if run["source_prefetch"] else "")
        key = (s.get("model_key"), scope, s.get("transport"), run["max_inflight"], s.get("measured_frames"))
        cases.setdefault(key, []).append(run)
    table = []
    for (model, profile, transport, inflight, frames), group in sorted(cases.items(), key=lambda kv: str(kv[0])):
        summaries = [run["summary"] for run in group]
        passed = [s for s in summaries if s.get("execution_status") == "PASS"]
        fps = [s["effective_fps"] for s in passed if s.get("effective_fps")]
        tracked_fps = [s["bytetrack_completion_fps"] for s in passed if s.get("bytetrack_completion_fps")]
        mean = statistics.fmean(fps) if fps else None
        cv = (statistics.stdev(fps) / mean) if len(fps) > 1 and mean else None
        stage_names = {name for s in passed for name in s.get("stages_ms", {})}
        stages = []
        for name in stage_names:
            p50 = [s["stages_ms"][name]["p50"] for s in passed
                   if name in s["stages_ms"] and s["stages_ms"][name].get("p50") is not None]
            p95 = [s["stages_ms"][name]["p95"] for s in passed
                   if name in s["stages_ms"] and s["stages_ms"][name].get("p95") is not None]
            if p50:
                stages.append({"stage": name, "p50_ms": statistics.fmean(p50),
                               "p95_ms": statistics.fmean(p95) if p95 else None})
        # Totals (call/RTT/e2e) wrap the other stages; rank component stages only.
        totals = {"adapter_call_ms", "adapter_reported_ms", "edge_adapter_call_ms",
                  "edge_adapter_reported_ms", "request_rtt_ms", "e2e_ms", "cloud_service_ms"}
        stages.sort(key=lambda row: (row["stage"] in totals, -row["p50_ms"]))
        if len(passed) < len(summaries) or not passed:
            gate = "FAIL"
        elif len(passed) < expected_reps:
            gate = "PROVISIONAL"
        elif cv is not None and cv > MAX_CV:
            gate = "UNSTABLE"
        else:
            gate = "PASS"

        def mean_of(field, scope):
            values = [run[scope][field] for run in group
                      if run.get(scope) and run[scope].get(field) is not None]
            return statistics.fmean(values) if values else None
        lane_rows = lanes(stages, mean)
        table.append({
            "lanes": lane_rows,
            "bottleneck": lane_rows[0]["lane"] if lane_rows else None,
            "model_key": model, "profile": profile, "transport": transport, "max_inflight": inflight,
            "measured_frames": frames, "repetitions": len(summaries), "passed": len(passed),
            "effective_fps_mean": mean, "effective_fps_cv": cv,
            "bytetrack_fps_mean": statistics.fmean(tracked_fps) if tracked_fps else None,
            "e2e_p95_ms": next((r["p95_ms"] for r in stages if r["stage"] == "e2e_ms"), None),
            "top_stages": [r for r in stages if r["stage"] not in totals][:3],
            "stages": stages,
            "edge_cpu_mean_pct": mean_of("host_cpu_mean_pct", "resources_local"),
            "edge_temp_max_c": max((run["resources_local"]["temp_max_c"] for run in group
                                    if (run.get("resources_local") or {}).get("temp_max_c") is not None), default=None),
            "local_gpu_util_mean_pct": mean_of("gpu_util_mean_pct", "resources_local"),
            "cloud_gpu_util_mean_pct": mean_of("gpu_util_mean_pct", "resources_cloud"),
            "cloud_gpu_mem_max_mib": mean_of("gpu_mem_max_mib", "resources_cloud"),
            "source_loop": any(run["source_loop"] for run in group),
            "gate": gate,
            "failures": sorted({s.get("fatal_error") or ",".join(s.get("failure_reasons", {}))
                                for s in summaries if s.get("execution_status") != "PASS"}),
        })
    return table


def _fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def write(table: list[dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Benchmark report", "",
             "| Model | Scope | Inflight | Frames | Reps ok | FPS mean | CV | ByteTrack FPS | e2e p95 ms | Top stages (p50 ms) | GPU % (Cloud) | CPU % (local) | Temp max C (local) | Gate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in table:
        top = ", ".join(f"{s['stage']} {s['p50_ms']:.1f}" for s in row["top_stages"]) or "-"
        gpu = row["cloud_gpu_util_mean_pct"] if row["cloud_gpu_util_mean_pct"] is not None else row["local_gpu_util_mean_pct"]
        cv = "-" if row["effective_fps_cv"] is None else f"{row['effective_fps_cv'] * 100:.1f}%"
        lines.append(f"| {row['model_key']} | {row['profile']} | {row['max_inflight'] or '-'} | {row['measured_frames']} "
                     f"| {row['passed']}/{row['repetitions']} | {_fmt(row['effective_fps_mean'])} | {cv} "
                     f"| {_fmt(row['bytetrack_fps_mean'])} | {_fmt(row['e2e_p95_ms'])} | {top} | {_fmt(gpu)} "
                     f"| {_fmt(row['edge_cpu_mean_pct'])} | {_fmt(row['edge_temp_max_c'])} | {row['gate']} |")
    lines += ["", "## Bottleneck (serial lanes, FPS cap = 1000 / lane p50 ms)", "",
              "| Model | Scope | Inflight | FPS mean | Lane | Stages | Busy p50 ms | FPS cap | Load |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in table:
        for index, lane in enumerate(row["lanes"]):
            name = f"**{lane['lane']}**" if index == 0 else lane["lane"]
            load = "-" if lane["load"] is None else f"{lane['load'] * 100:.0f}%"
            lines.append(f"| {row['model_key']} | {row['profile']} | {row['max_inflight'] or '-'} "
                         f"| {_fmt(row['effective_fps_mean'])} | {name} | {'+'.join(lane['stages'])} "
                         f"| {_fmt(lane['busy_p50_ms'], 2)} | {_fmt(lane['fps_cap'])} | {load} |")
    lines += ["", "Bold lane = lowest cap = bottleneck. Load near 100% confirms it limits FPS; "
              "low load on every lane means waiting (queue/network/inflight) dominates instead.",
              "", "Gate: PASS = all reps ok, >=3 reps, FPS CV <= 5%; PROVISIONAL = fewer reps; "
              "UNSTABLE = CV > 5%; FAIL = any rep failed.",
              "Content repeats when source_loop is true; capacity numbers, not quality."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "report.csv").open("w", encoding="utf-8", newline="") as target:
        fields = ["model_key", "profile", "transport", "max_inflight", "measured_frames", "repetitions",
                  "passed", "effective_fps_mean", "effective_fps_cv", "bytetrack_fps_mean", "e2e_p95_ms",
                  "edge_cpu_mean_pct", "edge_temp_max_c", "local_gpu_util_mean_pct", "cloud_gpu_util_mean_pct",
                  "cloud_gpu_mem_max_mib", "source_loop", "gate"]
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    (out / "report.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--cloud-resources", type=Path, help="Cloud receiver resources JSONL (same Cloud clock)")
    parser.add_argument("--expected-reps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    table = aggregate(load_runs(args.roots, args.cloud_resources), args.expected_reps)
    out = args.out or args.roots[0]
    write(table, out)
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0 if table and all(row["gate"] == "PASS" for row in table) else 1


if __name__ == "__main__":
    raise SystemExit(main())
