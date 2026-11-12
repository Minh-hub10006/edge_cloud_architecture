"""Slow-lane back-pressure (P1, 10/11): a Cloud worker that falls behind asks every Edge to space its slow-lane sends.

LagController sees the age of each message it starts (now - the frame's capture time) and, every `period_s`, sets a
throttle: x2 when the 95th percentile age is above `high_s`, /2 when below `low_s` (1 = normal, at most 8). Each worker
publishes {"source", "throttle", "lag_p95_s"} on cloud.slow-feedback.v1; an Edge uses the largest fresh throttle of
all sources (contracts/control_client_v1). Ages compare the Edge's clock with the Cloud's: a skew of a second or two
(seen on 247) is far below the thresholds.
"""
from __future__ import annotations

import json
import os
import socket
from time import monotonic, time_ns


class LagController:
    def __init__(self, source: str, *, high_s: float = 8.0, low_s: float = 2.0, period_s: float = 2.0,
                 max_throttle: float = 8.0) -> None:
        self.source = source
        self.high_s, self.low_s, self.period_s, self.max_throttle = high_s, low_s, period_s, max_throttle
        self.throttle, self.ages, self.next_at = 1.0, [], monotonic() + period_s

    def observe(self, capture_utc_ns: int, now_ns: int | None = None) -> float:
        age = ((now_ns or time_ns()) - capture_utc_ns) / 1e9
        self.ages.append(age)
        return age

    def tick(self, now: float | None = None) -> dict | None:
        """-> the feedback message to publish when a period ended, else None."""
        now = monotonic() if now is None else now
        if now < self.next_at:
            return None
        self.next_at = now + self.period_s
        ages, self.ages = sorted(self.ages), []
        p95 = ages[min(len(ages) - 1, int(.95 * len(ages)))] if ages else 0.0
        if p95 > self.high_s:
            self.throttle = min(self.max_throttle, self.throttle * 2)
        elif p95 < self.low_s:
            self.throttle = max(1.0, self.throttle / 2)
        return {"source": self.source, "throttle": self.throttle, "lag_p95_s": round(p95, 2), "messages": len(ages)}


def worker_source(kind: str) -> str:
    return f"{kind}@{socket.gethostname()}:{os.getpid()}"


def encode(message: dict) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode()
