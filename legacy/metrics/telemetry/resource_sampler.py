"""Low-overhead host resource sampling for benchmark telemetry."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class ResourceSampler:
    """Samples at a bounded interval; unavailable values stay None."""

    def __init__(self, *, interval_s: float = 1.0):
        self.interval_s = max(0.1, float(interval_s))
        self._next_sample_at = 0.0
        if psutil is not None:
            psutil.cpu_percent(interval=None)

    def sample_if_due(self) -> dict[str, Any] | None:
        now = time.monotonic()
        if now < self._next_sample_at:
            return None
        self._next_sample_at = now + self.interval_s
        row: dict[str, Any] = {
            "wall_time_ns": time.time_ns(),
            "cpu_utilization_ratio": None,
            "memory_utilization_ratio": None,
            "process_rss_bytes": None,
            "temperature_celsius": self._read_cpu_temperature(),
        }
        if psutil is not None:
            row["cpu_utilization_ratio"] = psutil.cpu_percent(interval=None) / 100.0
            row["memory_utilization_ratio"] = psutil.virtual_memory().percent / 100.0
            row["process_rss_bytes"] = psutil.Process().memory_info().rss
        return row

    @staticmethod
    def _read_cpu_temperature() -> float | None:
        candidates: list[float] = []
        for zone in Path("/sys/class/thermal").glob("thermal_zone*/"):
            try:
                kind = (zone / "type").read_text(encoding="utf-8").strip().lower()
                if "cpu" not in kind and "cpuss" not in kind:
                    continue
                raw = float((zone / "temp").read_text(encoding="utf-8").strip())
                value = raw / 1000.0 if raw > 1000.0 else raw
                if 0.0 < value < 150.0:
                    candidates.append(value)
            except (OSError, ValueError):
                continue
        return max(candidates) if candidates else None
