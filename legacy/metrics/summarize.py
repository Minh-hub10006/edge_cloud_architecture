#!/usr/bin/env python3
"""Summarize canonical JSONL traces into JSON and stage CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from benchmark.telemetry.trace import STAGES


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * (float(percent) / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "min": min(samples),
        "max": max(samples),
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status", "unknown")) for row in rows)
    good = [row for row in rows if row.get("status") == "ok"]
    run_ids = sorted({str(row.get("run_id", "")) for row in rows})
    requested_modes = Counter(str(row.get("mode", "unknown")) for row in rows)
    backends = Counter(str(row.get("backend", "unknown")) for row in rows)
    selected_modes = Counter(
        str((row.get("attributes") or {}).get("selected_mode", row.get("mode", "unknown")))
        for row in rows
    )
    transports = Counter(
        str((row.get("attributes") or {}).get("transport", _infer_transport(row)))
        for row in rows
    )

    starts = [int(row["wall_started_ns"]) for row in rows if row.get("wall_started_ns")]
    finishes = [int(row["wall_finished_ns"]) for row in rows if row.get("wall_finished_ns")]
    wall_time_s = (
        max(0.0, (max(finishes) - min(starts)) / 1_000_000_000.0)
        if starts and finishes
        else None
    )

    stage_summary: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        values = [
            float(row["durations_ms"][stage])
            for row in good
            if (row.get("durations_ms") or {}).get(stage) is not None
        ]
        stage_stats = stats(values)
        stage_stats["missing"] = len(good) - len(values)
        stage_summary[stage] = stage_stats

    payload_values = [
        float((row.get("attributes") or {}).get("payload_bytes"))
        for row in good
        if (row.get("attributes") or {}).get("payload_bytes") is not None
    ]
    transport_summary = {
        "modes": dict(transports),
        "request_envelope_bytes": stats(
            float((row.get("attributes") or {})["transport_request_envelope_bytes"])
            for row in good
            if (row.get("attributes") or {}).get("transport_request_envelope_bytes") is not None
        ),
        "result_envelope_bytes": stats(
            float((row.get("attributes") or {})["transport_result_envelope_bytes"])
            for row in good
            if (row.get("attributes") or {}).get("transport_result_envelope_bytes") is not None
        ),
        "runtime": _latest_transport_runtime(rows),
    }
    scene_summary = {
        key: stats(
            float((row.get("attributes") or {})[key])
            for row in good
            if (row.get("attributes") or {}).get(key) is not None
        )
        for key in ("box_count", "tracked_box_count", "confidence_mean", "box_change")
    }

    decision_frames = sum(
        bool((row.get("attributes") or {}).get("decision_made")) for row in rows
    )
    switch_frames = sum(
        bool((row.get("attributes") or {}).get("mode_changed")) for row in rows
    )

    return {
        "summary_version": 2,
        "run_ids": run_ids,
        "requested_modes": dict(requested_modes),
        "backends": dict(backends),
        "selected_modes": dict(selected_modes),
        "frames": {
            "total": len(rows),
            "completed": len(good),
            "timeout": statuses.get("timeout", 0),
            "dropped": statuses.get("dropped", 0),
            "error": statuses.get("error", 0),
            "other": sum(
                count
                for status, count in statuses.items()
                if status not in {"ok", "timeout", "dropped", "error"}
            ),
        },
        "wall_time_s": wall_time_s,
        "effective_fps": (
            len(good) / wall_time_s if wall_time_s is not None and wall_time_s > 0 else None
        ),
        "decision_frames": decision_frames,
        "model_switches": switch_frames,
        "payload_bytes": stats(payload_values),
        "transport": transport_summary,
        "scene_metrics": scene_summary,
        "stages_ms": stage_summary,
    }


def _infer_transport(row: dict[str, Any]) -> str:
    durations = row.get("durations_ms") or {}
    if durations.get("grpc_rtt") is not None:
        return "grpc"
    return "local"


def _latest_transport_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve a final bounded-client counter snapshot when supplied."""
    for row in reversed(rows):
        value = (row.get("attributes") or {}).get("transport_runtime")
        if isinstance(value, dict):
            return dict(value)
    return {}


def summarize_trace(path: str | Path) -> dict[str, Any]:
    return summarize_rows(load_jsonl(path))


def write_outputs(
    summary: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_out = Path(json_path)
    csv_out = Path(csv_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_out.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("stage", "count", "missing", "mean", "p50", "p95", "p99", "min", "max"),
        )
        writer.writeheader()
        for stage in STAGES:
            writer.writerow({"stage": stage, **summary["stages_ms"][stage]})


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSONL trace")
    parser.add_argument("trace")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()
    summary = summarize_trace(args.trace)
    write_outputs(summary, json_path=args.json_out, csv_path=args.csv_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
