"""Exact bounded percentile reports, inspired by plate_lisence benchmark."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from .config import RunConfig, finite_nonnegative


def stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(finite_nonnegative(x, "sample") for x in values)
    def percentile(percent: float):
        if not ordered:
            return None
        rank = (len(ordered) - 1) * percent / 100
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered) if ordered else None,
            "min": ordered[0] if ordered else None, "max": ordered[-1] if ordered else None,
            "p50": percentile(50), "p95": percentile(95), "p99": percentile(99)}


def summarize(path: Path, config: RunConfig, elapsed_s: float, *,
              fatal_error: str | None = None, identity_verified: bool = False) -> dict[str, Any]:
    elapsed = finite_nonnegative(elapsed_s, "elapsed_s")
    statuses: Counter[str] = Counter()
    stages: dict[str, list[float]] = {}
    failed_stages: dict[str, list[float]] = {}
    reasons: Counter[str] = Counter()
    seen = set()
    detections = warmup = warmup_errors = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            for key in ("campaign_id", "run_id", "node_id", "model_key", "placement",
                        "profile", "transport", "repetition"):
                if event.get(key) != getattr(config, key):
                    raise ValueError(f"mixed trace identity: {key}")
            if event.get("schema_version") != 1:
                raise ValueError("unknown trace schema")
            phase, status = event.get("phase"), event.get("status")
            if phase not in {"measure", "warmup"} or status not in {"ok", "error", "timeout", "dropped"}:
                raise ValueError("invalid trace status/phase")
            token = (phase, event["frame_id"])
            if token in seen:
                raise ValueError("duplicate frame observation")
            seen.add(token)
            if len(seen) > config.warmup_frames + config.measured_frames:
                raise ValueError("trace exceeds configured bound")
            if phase == "warmup":
                warmup += 1
                warmup_errors += status != "ok"
                continue
            statuses[status] += 1
            durations = event.get("durations_ms", {})
            for name in durations:
                stages.setdefault(name, [])
            if status == "ok":
                detections += event["detections"]
                for name, value in durations.items():
                    if value is not None:
                        stages[name].append(finite_nonnegative(value, name))
            else:
                reasons[str(event.get("error") or status)] += 1
                for name, value in durations.items():
                    if value is not None:
                        failed_stages.setdefault(name, []).append(finite_nonnegative(value, name))
    completed = statuses["ok"]
    counts_ok = (completed == config.measured_frames and sum(statuses.values()) == completed)
    if config.profile == "realtime":
        counts_ok = (completed > 0 and sum(statuses.values()) == config.measured_frames
                     and statuses["error"] == 0 and statuses["timeout"] == 0)
    execution = (counts_ok
                 and warmup == config.warmup_frames and not warmup_errors and fatal_error is None)
    measurement = fatal_error is None and sum(statuses.values()) == config.measured_frames and elapsed > 0
    return {"schema_version": 1, **config.as_dict(), "elapsed_s": elapsed,
            "completed": completed, "frames": dict(statuses), "warmup_observations": warmup,
            "warmup_errors": warmup_errors, "detections_total": detections,
            "effective_fps": completed / elapsed if elapsed > 0 else None,
            "stages_ms": {name: {**stats(values), "missing": completed - len(values)}
                          for name, values in stages.items()},
            "failed_stages_ms": {name: stats(values) for name, values in failed_stages.items()},
            "failure_reasons": dict(reasons),
            "drop_rate": statuses["dropped"] / config.measured_frames,
            "lossless_required": config.profile != "realtime",
            "gates": {"execution": execution, "measurement": measurement,
                      "artifact_runtime_identity": identity_verified, "quality": None},
            "execution_status": "PASS" if execution and measurement and identity_verified else "FAIL",
            "quality_status": "NOT_EVALUATED", "fatal_error": fatal_error,
            "warnings": ["small sample: tail percentiles unstable"] if completed < 100 else []}


def write_report(directory: Path, summary: dict[str, Any]) -> None:
    with (directory / "summary.json").open("x", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2, allow_nan=False)
    with (directory / "stages.csv").open("x", encoding="utf-8", newline="") as target:
        fields = ["stage", "count", "mean", "min", "max", "p50", "p95", "p99", "missing"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for name, data in summary["stages_ms"].items():
            writer.writerow({"stage": name, **data})


def coordinated_summary(summaries: list[dict[str, Any]], *, expected_nodes: list[str],
                        elapsed_s: float) -> dict[str, Any]:
    """Only call with one run/case and a coordinator-observed common interval."""
    elapsed = finite_nonnegative(elapsed_s, "coordinator elapsed")
    if elapsed <= 0 or not expected_nodes or len(set(expected_nodes)) != len(expected_nodes):
        raise ValueError("positive interval and unique expected nodes required")
    nodes = [row["node_id"] for row in summaries]
    if len(set(nodes)) != len(nodes) or not set(nodes) <= set(expected_nodes):
        raise ValueError("duplicate/unexpected nodes")
    identities = {(row["campaign_id"], row["run_id"], row["model_key"], row["placement"], row["profile"],
                   row["transport"], row["repetition"]) for row in summaries}
    if len(identities) > 1:
        raise ValueError("cannot aggregate different cases or intervals")
    missing = sorted(set(expected_nodes) - set(nodes))
    completed = sum(row["completed"] for row in summaries)
    counts = [row["completed"] for row in summaries]
    fairness = (sum(counts) ** 2 / (len(counts) * sum(n * n for n in counts))
                if counts and any(counts) else None)
    return {"elapsed_s": elapsed, "completed": completed,
            "coordinated_fps": completed / elapsed, "missing_nodes": missing,
            "completion_share_jain": fairness,
            "fairness_definition": "completed counts over common interval, NOT scheduler fairness proof",
            "status": "PASS" if not missing and all(row["execution_status"] == "PASS"
                                                        for row in summaries) else "INCOMPLETE",
            "nodes": summaries, "interval_definition": "coordinator START to all DONE"}
