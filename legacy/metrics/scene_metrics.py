"""Model-neutral scene metrics used by the benchmark and DRL dataset."""
from __future__ import annotations

import math
import time
from typing import Any


def boxes_to_pixels(boxes: list[dict[str, Any]] | None, frame: Any, coordinate_space=None) -> list[dict[str, Any]]:
    """Copy valid boxes into source-frame pixel coordinates."""
    # Validate raw values before clipping: min/max can turn NaN into an edge.
    parsed = []
    for original in boxes or []:
        try:
            box = dict(original)
            coords = [float(box[key]) for key in ("left", "top", "right", "bottom")]
            confidence = float(box.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(v) for v in (*coords, confidence)):
            continue
        parsed.append((box, coords, confidence))
    if not parsed:
        return []
    normalized = coordinate_space == "normalized" or (
        coordinate_space is None
        and max(abs(v) for _, coords, _ in parsed for v in coords) <= 1.5
    )
    height, width = frame.shape[:2] if frame is not None else (None, None)
    valid = []
    for box, coords, confidence in parsed:
        left, top, right, bottom = coords
        if width is not None:
            if normalized:
                left, right = left * width, right * width
                top, bottom = top * height, bottom * height
            if not all(math.isfinite(v) for v in (left, top, right, bottom)):
                continue
            left, right = max(0.0, min(width, left)), max(0.0, min(width, right))
            top, bottom = max(0.0, min(height, top)), max(0.0, min(height, bottom))
        if right - left <= 1e-6 or bottom - top <= 1e-6:
            continue
        box.update(left=left, top=top, right=right, bottom=bottom)
        if "confidence" in box:
            box["confidence"] = confidence
        valid.append(box)
    return valid


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    left, top = max(float(a["left"]), float(b["left"])), max(float(a["top"]), float(b["top"]))
    right, bottom = min(float(a["right"]), float(b["right"])), min(float(a["bottom"]), float(b["bottom"]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a["right"]) - float(a["left"])) * max(0.0, float(a["bottom"]) - float(a["top"]))
    area_b = max(0.0, float(b["right"]) - float(b["left"])) * max(0.0, float(b["bottom"]) - float(b["top"]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def track_box_change(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> float:
    """DRL V2 temporal score: half ID churn and half tracked-box motion."""
    prev = {int(box["track_id"]): box for box in previous if int(box.get("track_id", -1)) >= 0}
    curr = {int(box["track_id"]): box for box in current if int(box.get("track_id", -1)) >= 0}
    all_ids, common = set(prev) | set(curr), set(prev) & set(curr)
    if not all_ids:
        return 0.0
    churn = 1.0 - len(common) / len(all_ids)
    motion = 1.0 - sum(_iou(prev[key], curr[key]) for key in common) / len(common) if common else 1.0
    return max(0.0, min(1.0, 0.5 * churn + 0.5 * motion))


class SceneMetricAccumulator:
    """Stateful ByteTrack metrics; create one instance per benchmark run."""
    def __init__(self, frame_rate: int = 30, tracking: bool = True):
        self._tracking = bool(tracking)
        if self._tracking:
            from trackers.byte_track import DEFAULT_TRACK_CLASSES, EdgeTracker

            self._tracker = EdgeTracker(frame_rate=frame_rate)
            self._keep_classes = DEFAULT_TRACK_CLASSES
        else:
            self._tracker = None
            self._keep_classes = ()
        self._previous: list[dict[str, Any]] = []
    @property
    def tracking_enabled(self) -> bool:
        return self._tracking

    def _summarize(self, pixels, tracked):
        scores = [float(box.get("confidence", 0.0)) for box in pixels]
        change = track_box_change(self._previous, tracked) if self._tracking else None
        self._previous = [dict(box) for box in tracked]
        return {
            "box_count": len(pixels),
            "tracked_box_count": sum(int(box.get("track_id", -1)) >= 0 for box in tracked) if self._tracking else None,
            "confidence_mean": sum(scores) / len(scores) if scores else 0.0,
            "box_change": change,
            "scene_metric_source": "bytetrack" if self._tracking else "detector_only",
        }

    def observe(self, boxes, frame, coordinate_space=None):
        pixels = boxes_to_pixels(boxes, frame, coordinate_space)
        tracked = self._tracker.update(pixels, frame, keep_classes=self._keep_classes) if self._tracking else []
        return self._summarize(pixels, tracked)

    def observe_traced(self, boxes, frame, trace, coordinate_space=None):
        metrics_started = time.perf_counter()
        pixels = boxes_to_pixels(boxes, frame, coordinate_space)
        metrics_s = time.perf_counter() - metrics_started
        if self._tracking:
            with trace.span("tracking"):
                tracked = self._tracker.update(pixels, frame, keep_classes=self._keep_classes)
        else:
            tracked = []
        metrics_started = time.perf_counter()
        result = self._summarize(pixels, tracked)
        metrics_s += time.perf_counter() - metrics_started
        trace.observe_ms("scene_metrics", metrics_s * 1000.0)
        return result
