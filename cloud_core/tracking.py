"""Optional Ultralytics ByteTrack adapter for normalized platform results."""

from __future__ import annotations

import json
import os
from threading import RLock
from types import SimpleNamespace
from typing import Any

from contracts import Box, DetectionResult


_CANONICAL_SIZE = 1024


class _DetectionContainer:
    """Small Ultralytics-compatible view over xyxy/conf/class rows."""

    def __init__(self, rows: Any) -> None:
        import numpy as np

        self.rows = np.asarray(rows, dtype=np.float32).reshape((-1, 6))
        xyxy = self.rows[:, :4]
        self._xywh = np.empty((len(self.rows), 4), dtype=np.float32)
        if len(self.rows):
            self._xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
            self._xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
            self._xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
            self._xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    # NumPy, as Ultralytics passes boxes.cpu().numpy() to trackers: a size-1
    # torch bool mask indexing a NumPy array is read as integer index 1.
    @property
    def conf(self):
        return self.rows[:, 4]

    @property
    def xywh(self):
        return self._xywh

    @property
    def cls(self):
        return self.rows[:, 5].astype("int64")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np

        if hasattr(index, "cpu"):
            index = index.cpu().numpy()
        return _DetectionContainer(np.atleast_2d(self.rows[index]))


_VEHICLE_CLASSES = frozenset({2, 5, 7})  # COCO car, bus, truck


def merge_vehicle_duplicates(boxes, iou_threshold: float):
    """Keep the higher-score box when car/bus/truck labels overlap one vehicle."""

    def iou(a, b):
        w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
        h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
        inter = w * h
        union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
        return inter / union if union > 0 else 0.0

    kept = []
    for box in sorted(boxes, key=lambda item: -item.confidence):
        if box.class_id in _VEHICLE_CLASSES and any(
                other.class_id in _VEHICLE_CLASSES and other.class_id != box.class_id
                and iou(box, other) >= iou_threshold for other in kept):
            continue
        kept.append(box)
    # Preserve detector order for the boxes that remain.
    survivors = {id(box) for box in kept}
    return [box for box in boxes if id(box) in survivors]


class _StreamByteTrack:
    def __init__(self, frame_rate: int, options: dict[str, Any],
                 vehicle_merge_iou: float | None = None,
                 class_allowlist: frozenset[int] | None = None) -> None:
        import numpy as np
        from ultralytics.trackers import BYTETracker

        self.vehicle_merge_iou = vehicle_merge_iou
        self.class_allowlist = class_allowlist
        args = SimpleNamespace(**options)
        try:
            self.tracker = BYTETracker(args, frame_rate=frame_rate)
        except TypeError:  # Compatibility with older Ultralytics releases.
            self.tracker = BYTETracker(args)
        # BYTETracker uses only the shape from this image; detections are kept
        # in a fixed normalized coordinate plane, so no source pixels are needed.
        self.coordinate_plane = np.empty((_CANONICAL_SIZE, _CANONICAL_SIZE, 3), dtype=np.uint8)
        self.frame_rate = frame_rate
        self.last_tick = None

    def update(self, result: DetectionResult) -> DetectionResult:
        import numpy as np
        from dataclasses import replace

        if self.class_allowlist is not None:
            kept = tuple(box for box in result.boxes if box.class_id in self.class_allowlist)
            result = replace(result, boxes=kept, attributes={
                **result.attributes, "class_filtered": len(result.boxes) - len(kept)})
        if self.vehicle_merge_iou is not None:
            merged = merge_vehicle_duplicates(result.boxes, self.vehicle_merge_iou)
            result = replace(result, boxes=tuple(merged), attributes={
                **result.attributes, "vehicle_duplicates_merged": len(result.boxes) - len(merged)})
        rows = np.asarray([
            [
                box.left * _CANONICAL_SIZE,
                box.top * _CANONICAL_SIZE,
                box.right * _CANONICAL_SIZE,
                box.bottom * _CANONICAL_SIZE,
                box.confidence,
                box.class_id,
            ]
            for box in result.boxes
        ], dtype=np.float32).reshape((-1, 6))
        # Advance missing time steps instead of interpreting every result as
        # the next adjacent video frame. Long gaps reset rather than extrapolate.
        pts = result.preview_pts if result.preview_pts is not None else result.frame.source_pts
        tick = round(pts * self.frame_rate / 1e9)
        if self.last_tick is not None:
            gap = max(1, tick - self.last_tick)
            if gap > self.frame_rate * 2:
                self.tracker.reset()
            else:
                empty = _DetectionContainer(np.empty((0, 6), dtype=np.float32))
                for _ in range(gap - 1):
                    self.tracker.update(empty, self.coordinate_plane)
        self.last_tick = tick
        tracked = self.tracker.update(_DetectionContainer(rows), self.coordinate_plane)
        track_by_detection: dict[int, int] = {}
        if tracked is not None:
            for row in tracked:
                # Ultralytics ByteTrack returns orig detection index in col 7.
                if len(row) >= 8 and int(row[7]) >= 0:
                    track_by_detection[int(row[7])] = int(row[4])
        boxes = tuple(
            replace(box, track_id=track_by_detection.get(index))
            for index, box in enumerate(result.boxes)
        )
        return replace(result, boxes=boxes)


class CloudByteTrack:
    """Lazily create independent ByteTrack state per stream epoch.

    Install the optional ``ultralytics``, ``numpy`` and ``torch`` packages on
    the Cloud worker before constructing this adapter. Imports are lazy so the
    contract-only platform remains importable without model dependencies.
    """

    def __init__(
        self, *, frame_rate: int = 15, track_high_thresh: float = 0.3,
        track_low_thresh: float = 0.1, new_track_thresh: float = 0.6,
        track_buffer: int = 60, match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        # 8/11 (scripts/tracker_eval.py, RT-DETRv2-S boxes >= .5 at 15 fps): match_thresh .5 with score fusion needed
        # IoU x score >= .5, so most boxes failed to match and got a new ID every frame (VN median track life 1 frame,
        # DETRAC 2.7-6.1 IDs per vehicle). match .8 + new_track .6: DETRAC ID switches 43/388/331 -> 9/123/65,
        # 1.1-1.7 IDs per vehicle; VN median life 30-39 frames. Low-score boxes (< .5) fed to it made it worse.
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.frame_rate = frame_rate
        self.options = {
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "fuse_score": fuse_score,
        }
        # Opt-in per-receiver override (JSON object); baseline defaults stay unchanged.
        override = os.getenv("CLOUD_BYTETRACK_OPTIONS", "").strip()
        if override:
            values = json.loads(override)
            if not isinstance(values, dict) or set(values) - set(self.options):
                raise ValueError("CLOUD_BYTETRACK_OPTIONS must be a JSON object of known ByteTrack options")
            self.options.update(values)
        # Opt-in: merge cross-class car/bus/truck boxes on one vehicle before tracking.
        merge = os.getenv("CLOUD_VEHICLE_MERGE_IOU", "").strip()
        self.vehicle_merge_iou = float(merge) if merge else None
        if self.vehicle_merge_iou is not None and not 0 < self.vehicle_merge_iou <= 1:
            raise ValueError("CLOUD_VEHICLE_MERGE_IOU must be in (0, 1]")
        # Opt-in: drop classes outside the traffic domain (e.g. a building seen as "train").
        allow = os.getenv("CLOUD_CLASS_ALLOWLIST", "").strip()
        self.class_allowlist = frozenset(int(item) for item in allow.split(",")) if allow else None
        if self.class_allowlist is not None and not all(0 <= item < 80 for item in self.class_allowlist):
            raise ValueError("CLOUD_CLASS_ALLOWLIST must be COCO80 class ids")
        self._streams: dict[tuple[str, str, str, int], _StreamByteTrack] = {}
        self._lock = RLock()

    def update(self, result: DetectionResult) -> DetectionResult:
        key = result.frame.stream_key
        with self._lock:
            tracker = self._streams.get(key)
            if tracker is None:
                rate = int(result.attributes.get("media_fps", self.frame_rate))
                tracker = _StreamByteTrack(rate if 0 < rate <= 240 else self.frame_rate, self.options,
                                           self.vehicle_merge_iou, self.class_allowlist)
                self._streams[key] = tracker
        return tracker.update(result)

    def prepare(self) -> None:
        """Load optional tracker dependencies before accepting real deadlines."""
        _StreamByteTrack(self.frame_rate, self.options)

    def reset(self, stream_key: tuple[str, str, str, int]) -> None:
        with self._lock:
            self._streams.pop(stream_key, None)

    @property
    def stream_count(self) -> int:
        with self._lock:
            return len(self._streams)
