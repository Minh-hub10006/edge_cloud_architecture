"""Slow lane tasks, step 2 (8/11): UC3 counting and UC2 congestion, from the fast lane's tracked boxes (no pixels).

Both work on fixed windows (default 5 s, decision slow-lane-v4: fixed low rates, not real time except alerts) and
send what they find through `sink(kind, payload)`: kind "count" -> table counts, kind "event" -> table events (and is
the alert: an event is sent the moment it starts, and again when it ends).

Thresholds are per camera (--slow-config JSON); defaults agreed 8/11: congestion = at least 30 vehicles, at least 70 %
of the measurable ones standing (moving < 0.2 box heights per second), for 30 s; it ends after 30 s without that.
"""
from __future__ import annotations

from collections import OrderedDict, deque

from .camera_profile import in_zone
from .events import Episode
from .slow_lane import next_event

CLASSES = {0: "persons", 1: "bicycles", 2: "cars", 3: "motorcycles", 5: "buses", 7: "trucks"}  # COCO ids
VEHICLES = frozenset({2, 3, 5, 7})
DEFAULTS = {"window_s": 5.0, "congestion_vehicles": 30, "congestion_stationary_share": 0.7, "congestion_hold_s": 30.0,
            "stationary_speed": 0.2}


def stream_fields(frame) -> dict:
    i = frame.identity
    return {"node_id": i.node_id, "camera_id": i.camera_id, "stream_id": i.stream_id, "stream_epoch": i.stream_epoch}


class TrackMotion:
    """Speed of each tracked box in box heights per second, over the last ~1-2 s of its centre positions."""

    def __init__(self, keep: int = 2000) -> None:
        self.history: OrderedDict[int, deque] = OrderedDict()
        self.keep = keep

    def update(self, track_id: int, t_ns: int, box) -> float | None:
        cx, cy, h = (box.left + box.right) / 2, (box.top + box.bottom) / 2, max(box.bottom - box.top, 1e-3)
        past = self.history.pop(track_id, None) or deque()
        past.append((t_ns, cx, cy))
        while past and t_ns - past[0][0] > 2_000_000_000:
            past.popleft()
        self.history[track_id] = past
        if len(self.history) > self.keep:
            self.history.popitem(last=False)
        t0, x0, y0 = past[0]
        span = (t_ns - t0) / 1e9
        if span < .5:
            return None   # not long enough to tell standing from moving
        return ((cx - x0) ** 2 + (cy - y0) ** 2) ** .5 / h / span


class Windows:
    """Fixed windows on the frames' own clock: add(frame) returns True when the frame closed the current window."""

    def __init__(self, seconds: float) -> None:
        self.span_ns, self.start_ns = round(seconds * 1e9), None

    def closes(self, t_ns: int) -> bool:
        if self.start_ns is None:
            self.start_ns = t_ns
        if t_ns - self.start_ns >= self.span_ns:
            self.start_ns = t_ns
            return True
        return False


class CountTask:
    """UC3 (sparse scenes): people and vehicles counted from the boxes, per window -> one counts row."""

    name = "count"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    async def run(self, queue, ring, stop) -> dict:
        windows, motion, seen = Windows(self.cfg["window_s"]), TrackMotion(), OrderedDict()
        area = (self.cfg.get("zones") or {}).get("area")
        acc, sent = self._empty(), 0
        last = None
        while (event := await next_event(queue, stop)) is not None:
            last = event.frame
            t = last.capture_mono_ns
            per = dict.fromkeys(CLASSES.values(), 0)
            standing = measured = 0
            for box in event.result.boxes:
                if not in_zone(box, area):
                    continue
                if box.class_id in CLASSES:
                    per[CLASSES[box.class_id]] += 1
                if box.track_id is not None:
                    if box.track_id not in seen:
                        acc["new_tracks"] += 1
                    seen[box.track_id] = None
                    seen.move_to_end(box.track_id)
                    if len(seen) > 5000:
                        seen.popitem(last=False)
                    if box.class_id in VEHICLES:
                        speed = motion.update(box.track_id, t, box)
                        if speed is not None:
                            measured += 1
                            standing += speed < self.cfg["stationary_speed"]
            acc["frames"] += 1
            for key, value in per.items():
                acc[key] += value
            vehicles = sum(per[CLASSES[c]] for c in VEHICLES)
            acc["vehicles_max"] = max(acc["vehicles_max"], vehicles)
            acc["persons_max"] = max(acc["persons_max"], per["persons"])
            acc["standing"] += standing
            acc["measured"] += measured
            if windows.closes(t):
                sent += await self._send(acc, last)
                acc = self._empty()
        if last is not None and acc["frames"]:
            sent += await self._send(acc, last)
        return {"windows_sent": sent}

    @staticmethod
    def _empty():
        return {"frames": 0, "new_tracks": 0, "vehicles_max": 0, "persons_max": 0, "standing": 0, "measured": 0,
                **dict.fromkeys(CLASSES.values(), 0)}

    async def _send(self, acc, frame) -> int:
        n = acc["frames"]
        payload = {**stream_fields(frame), "window_end_utc_ns": frame.identity.capture_utc_ns,
                   "window_s": self.cfg["window_s"], "frames": n,
                   **{key: round(acc[key] / n, 2) for key in CLASSES.values()},   # mean per analysed frame
                   "vehicles_max": acc["vehicles_max"], "persons_max": acc["persons_max"],
                   "new_tracks": acc["new_tracks"],
                   "stationary_share": round(acc["standing"] / acc["measured"], 3) if acc["measured"] else None}
        if self.sink is not None:
            await self.sink("count", payload)
        return 1


class CongestionTask:
    """UC2 congestion: windows where vehicles >= N and most of them stand; starts after hold_s of such windows, ends
    after hold_s without. One event per episode (start sent at once = the alert, end updates the same event)."""

    name = "congestion"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    async def run(self, queue, ring, stop) -> dict:
        cfg = self.cfg
        hold = max(1, round(cfg["congestion_hold_s"] / cfg["window_s"]))
        windows, motion = Windows(cfg["window_s"]), TrackMotion()
        road = (cfg.get("zones") or {}).get("road")
        vehicles = frames = standing = measured = 0
        episode = Episode("congestion", hold, hold)
        while (event := await next_event(queue, stop)) is not None:
            frame = event.frame
            t = frame.capture_mono_ns
            boxes = [b for b in event.result.boxes if b.class_id in VEHICLES and in_zone(b, road)]
            vehicles += len(boxes)
            frames += 1
            for box in boxes:
                if box.track_id is not None and (speed := motion.update(box.track_id, t, box)) is not None:
                    measured += 1
                    standing += speed < cfg["stationary_speed"]
            if not windows.closes(t):
                continue
            mean = vehicles / frames
            share = standing / measured if measured else 0.0
            jammed = mean >= cfg["congestion_vehicles"] and share >= cfg["congestion_stationary_share"]
            vehicles = frames = standing = measured = 0
            detail = {"vehicles_mean": round(mean, 1), "stationary_share": round(share, 3)}
            if (message := episode.update(jammed, frame, detail)) is not None:
                await self._send(message)
        return {"events_started": episode.started, "open_at_end": episode.active is not None}

    async def _send(self, payload):
        if self.sink is not None:
            await self.sink("event", payload)
