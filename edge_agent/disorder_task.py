"""UC4 on the Edge (P3, 11/11): rules only, no model; the Cloud confirms with RTMO-s (+ ST-GCN++ for fights).

fight   people on foot (not riding: their box is not on a motorbike/bicycle) standing close (ground points within
        `fight_gap` of their box height), at least one agitated: over the last second its box centre travelled
        >= `fight_speed` box heights/s while going nowhere (path >= `fight_back_forth` x net move; people walking
        together go straight, ratio ~1) -> a short clip:
        `clip_frames` crops around the group, `clip_every_s` apart, from the 3 s ring (full resolution where the ring
        still has it, else the analysis frame) -> Cloud (cloud_core/uc4_disorder.py: RTMO-s + ST-GCN++). At most one
        clip per `fight_every_s` x throttle per camera.
crash   a motorbike/bicycle track that was moving (>= `crash_moving_speed`) in the last `crash_window_s`, is now
        stopped (< `crash_stop_speed`) and lying (box wider than `crash_lying_ratio` x its height, in pixels) for
        `crash_hold_s` -> one crop around it -> Cloud (RTMO-s: is someone lying next to it?). Once per track.
loiter  a person (zone "watch") tracked for `loiter_s` who stayed within `loiter_radius` box heights of where they were
        first seen -> event "loitering" (start; end when the track is gone `loiter_gone_s`). Edge only.
11/11 measured on public clips: NTU60 action models alone do not transfer (fight AUC 0.44-0.70), hence rules first and
the action model only on what the rules flag.
"""
from __future__ import annotations

import base64
from collections import deque

import numpy as np

from .camera_profile import in_zone
from .events import Episode
from .slow_lane import next_event
from .slow_tasks import TrackMotion, stream_fields

PERSON, BICYCLE, MOTORBIKE = 0, 1, 3
DEFAULTS = {"fight_gap": 1.0, "fight_speed": .8, "fight_back_forth": 2.0, "fight_persons": 2, "fight_every_s": 10.0, "clip_frames": 10,
            "clip_every_s": .2, "clip_long_side": 512, "clip_jpeg_quality": 80, "action_deadline_s": 15.0,
            "crash_moving_speed": .5, "crash_stop_speed": .15, "crash_lying_ratio": 1.3, "crash_hold_s": 2.0,
            "crash_window_s": 4.0, "loiter_s": 120.0, "loiter_radius": 3.0, "loiter_gone_s": 10.0}


class Agitation:
    """Per track, over the last `seconds`: path travelled by the box centre and the net move, in box heights/s."""

    def __init__(self, seconds: float = 1.0, keep: int = 2000) -> None:
        self.span_ns, self.keep, self.history = round(seconds * 1e9), keep, {}

    def update(self, track_id: int, t_ns: int, box) -> tuple[float, float] | None:
        past = self.history.pop(track_id, None) or deque()
        past.append((t_ns, (box.left + box.right) / 2, (box.top + box.bottom) / 2, max(box.bottom - box.top, 1e-3)))
        while t_ns - past[0][0] > self.span_ns:
            past.popleft()
        self.history[track_id] = past
        if len(self.history) > self.keep:
            self.history.pop(next(iter(self.history)))
        span = (t_ns - past[0][0]) / 1e9
        if span < .5 or len(past) < 3:
            return None
        h = sum(p[3] for p in past) / len(past)
        path = sum(((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** .5 for a, b in zip(past, list(past)[1:]))
        net = ((past[-1][1] - past[0][1]) ** 2 + (past[-1][2] - past[0][2]) ** 2) ** .5
        return path / h / span, net / h / span


def frame_size(frame) -> tuple[int, int]:
    """(w, h) of the camera picture the normalized boxes refer to."""
    size = getattr(frame, "source_size", None)
    if size is not None:
        return int(size[0]), int(size[1])
    h, w = frame.pixels.shape[:2]
    return w, h


def riding(person, bikes) -> bool:
    """The person's ground point lies inside a bike box (a rider, not someone on foot)."""
    x, y = (person.left + person.right) / 2, person.bottom
    return any(b.left <= x <= b.right and b.top <= y <= b.bottom + .5 * (b.bottom - b.top) for b in bikes)


def close_group(people, gap: float) -> list:
    """The largest set of people linked by ground distance <= gap x mean box height (union of pairs)."""
    groups = [{i} for i in range(len(people))]
    for i, a in enumerate(people):
        for j in range(i + 1, len(people)):
            b = people[j]
            d = (((a.left + a.right) - (b.left + b.right)) ** 2 / 4 + (a.bottom - b.bottom) ** 2) ** .5
            if d <= gap * ((a.bottom - a.top) + (b.bottom - b.top)) / 2:
                gi = next(g for g in groups if i in g)
                gj = next(g for g in groups if j in g)
                if gi is not gj:
                    gi |= gj
                    groups.remove(gj)
    best = max(groups, key=len, default=set())
    return [people[i] for i in sorted(best)]


def crop_region(boxes, margin: float = .5) -> tuple[float, float, float, float]:
    """Normalized (left, top, right, bottom) around the boxes, grown by `margin` of their size on each side."""
    left, top = min(b.left for b in boxes), min(b.top for b in boxes)
    right, bottom = max(b.right for b in boxes), max(b.bottom for b in boxes)
    mw, mh = (right - left) * margin, (bottom - top) * margin
    return max(0.0, left - mw), max(0.0, top - mh), min(1.0, right + mw), min(1.0, bottom + mh)


def crop_jpeg(frame, ring, region, long_side: int, quality: int, out_wh=None):
    """Crop `region` from the frame's full-resolution copy (own, or the ring's for this frame id), else from the analysis
    frame (stretched: normalized coords still hold) -> (base64 JPEG, (w, h), full resolution?). out_wh: force the size
    (every frame of one clip the same size, as the pose -> action input needs)."""
    import cv2
    src = frame if getattr(frame, "hires", None) is not None else ring.hires_near(frame.identity.source_frame_id, 0)
    img = src.hires if src is not None else frame.pixels
    h, w = img.shape[:2]
    l, t, r, b = region
    x0, y0, x1, y1 = int(l * w), int(t * h), max(int(l * w) + 1, int(r * w)), max(int(t * h) + 1, int(b * h))
    crop = img[y0:y1, x0:x1]
    if out_wh is None:
        cw, ch = frame_size(frame)
        rw, rh = (r - l) * cw, (b - t) * ch            # the region in camera pixels (true aspect)
        s = min(1.0, long_side / max(rw, rh, 1))
        out_wh = (max(1, round(rw * s)), max(1, round(rh * s)))
    crop = cv2.resize(crop, out_wh, interpolation=cv2.INTER_AREA if crop.shape[1] > out_wh[0] else cv2.INTER_LINEAR)
    ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (base64.b64encode(jpeg.tobytes()).decode() if ok else None), out_wh, src is not None


class FightTask:
    name = "fight"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    async def run(self, queue, ring, stop) -> dict:
        cfg = self.cfg
        throttle = cfg.get("throttle") or (lambda: 1.0)
        watch = (cfg.get("zones") or {}).get("watch")
        motion, next_ns = Agitation(), 0
        stats = {"suspect_frames": 0, "clips": 0, "clip_frames_low_res": 0}
        while (event := await next_event(queue, stop)) is not None:
            frame, boxes = event.frame, event.result.boxes
            t = frame.capture_mono_ns
            bikes = [b for b in boxes if b.class_id in (BICYCLE, MOTORBIKE)]
            fast = []
            for b in boxes:
                if b.class_id != PERSON or b.track_id is None or not in_zone(b, watch) or riding(b, bikes):
                    continue
                m = motion.update(b.track_id, t, b)
                if m is not None:
                    path, net = m
                    fast.append((b, path >= cfg["fight_speed"] and path >= cfg["fight_back_forth"] * net))
            group = close_group([b for b, _ in fast], cfg["fight_gap"])
            agitated = {id(b) for b, a in fast if a}
            if len(group) < cfg["fight_persons"] or not any(id(b) in agitated for b in group):
                continue
            stats["suspect_frames"] += 1
            if t < next_ns:
                continue
            next_ns = t + round(cfg["fight_every_s"] * throttle() * 1e9)
            region = crop_region(group)
            clip, size = [], None
            for f in ring.sequence(int(cfg["clip_frames"]), round(cfg["clip_every_s"] * 1e9)):
                b64, size, full = crop_jpeg(f, ring, region, int(cfg["clip_long_side"]), int(cfg["clip_jpeg_quality"]),
                                            size)
                if b64 is not None:
                    stats["clip_frames_low_res"] += not full
                    clip.append({"frame_id": f.identity.source_frame_id, "capture_utc_ns": f.identity.capture_utc_ns,
                                 "jpeg_b64": b64})
            if len(clip) < 2:
                continue
            stats["clips"] += 1
            capture = frame.identity.capture_utc_ns
            if self.sink is not None:
                await self.sink("action", {**stream_fields(frame), "frame_id": frame.identity.source_frame_id,
                                           "capture_utc_ns": capture,
                                           "deadline_utc_ns": capture + round(cfg["action_deadline_s"] * 1e9),
                                           "check": "fight", "region": [round(v, 4) for v in region],
                                           "persons": len(group), "tracks": [b.track_id for b in group],
                                           "width": size[0], "height": size[1], "frames": clip})
        return stats


class CrashTask:
    name = "crash"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    async def run(self, queue, ring, stop) -> dict:
        cfg = self.cfg
        road = (cfg.get("zones") or {}).get("road")
        motion = TrackMotion()
        moving_at: dict[int, int] = {}     # track -> last time it moved
        lying_since: dict[int, int] = {}   # track -> since when it is stopped and lying
        sent: set[int] = set()
        stats = {"suspects": 0, "sent": 0}
        while (event := await next_event(queue, stop)) is not None:
            frame = event.frame
            t = frame.capture_mono_ns
            w, h = frame_size(frame)
            for b in event.result.boxes:
                if b.class_id not in (BICYCLE, MOTORBIKE) or b.track_id is None or not in_zone(b, road):
                    continue
                k, speed = b.track_id, motion.update(b.track_id, t, b)
                if speed is None:
                    continue
                if speed >= cfg["crash_moving_speed"]:
                    moving_at[k] = t
                lying = (b.right - b.left) * w >= cfg["crash_lying_ratio"] * (b.bottom - b.top) * h
                recent = t - moving_at.get(k, -10 ** 18) <= cfg["crash_window_s"] * 1e9
                if not (lying and speed < cfg["crash_stop_speed"] and (recent or k in lying_since)):
                    lying_since.pop(k, None)
                    continue
                lying_since.setdefault(k, t)
                if t - lying_since[k] < cfg["crash_hold_s"] * 1e9 or k in sent:
                    continue
                sent.add(k)
                stats["suspects"] += 1
                b64, (cw, ch), full = crop_jpeg(frame, ring, crop_region([b], 1.0), 640, 85)
                if b64 is None:
                    continue
                stats["sent"] += 1
                capture = frame.identity.capture_utc_ns
                if self.sink is not None:
                    await self.sink("crash", {**stream_fields(frame), "frame_id": frame.identity.source_frame_id,
                                              "capture_utc_ns": capture,
                                              "deadline_utc_ns": capture + round(cfg["action_deadline_s"] * 1e9),
                                              "track_id": k, "box": [b.left, b.top, b.right, b.bottom],
                                              "width": cw, "height": ch, "full_res": full, "jpeg_b64": b64})
            if len(moving_at) > 2000:   # bounded: forget the oldest tracks
                for k in sorted(moving_at, key=moving_at.get)[:1000]:
                    moving_at.pop(k, None)
                    lying_since.pop(k, None)
        return stats


class LoiterTask:
    name = "loiter"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    async def run(self, queue, ring, stop) -> dict:
        cfg = self.cfg
        watch = (cfg.get("zones") or {}).get("watch")
        first: dict[int, tuple] = {}       # track -> (t, x, y) when first seen
        last_seen: dict[int, int] = {}
        open_: dict[int, Episode] = {}
        stats = {"started": 0}
        while (event := await next_event(queue, stop)) is not None:
            frame = event.frame
            t = frame.capture_mono_ns
            for b in event.result.boxes:
                if b.class_id != PERSON or b.track_id is None or not in_zone(b, watch):
                    continue
                k, x, y, bh = b.track_id, (b.left + b.right) / 2, b.bottom, max(b.bottom - b.top, 1e-3)
                t0, x0, y0 = first.setdefault(k, (t, x, y))
                last_seen[k] = t
                if ((x - x0) ** 2 + (y - y0) ** 2) ** .5 > cfg["loiter_radius"] * bh:
                    if k not in open_:
                        first[k] = (t, x, y)   # moved away: the clock restarts here
                    continue
                if k not in open_ and t - t0 >= cfg["loiter_s"] * 1e9:
                    open_[k] = Episode("loitering", 1, 1)
                    message = open_[k].update(True, frame, {"track_id": k, "seconds": round((t - t0) / 1e9)}, str(k))
                    stats["started"] += 1
                    if self.sink is not None:
                        await self.sink("event", message)
            for k in [k for k, s in last_seen.items() if t - s > cfg["loiter_gone_s"] * 1e9]:
                episode = open_.pop(k, None)
                if episode is not None and self.sink is not None:
                    await self.sink("event", episode.update(False, frame, {"track_id": k, "reason": "gone"}, str(k)))
                first.pop(k, None)
                last_seen.pop(k, None)
        return stats
