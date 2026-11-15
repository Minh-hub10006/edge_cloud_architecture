"""UC3 on the Edge (P2, 10/11): rules only, no model.

crowd      when the fast lane's person boxes (zone "area") say the scene is dense - at least `crowd_persons` people, or at
           least `crowd_overlap_share` of them overlapping another person (boxes merge: the detector undercounts) -
           send one full-resolution frame (long side 1280, JPEG) every `crowd_every_s` x throttle to the Cloud, where
           DM-Count estimates the count and where the people are (cloud_core/uc3_crowd.py).
gathering  people standing close together (zone "watch"): a group of at least `gathering_persons` tracked people, each
           within `gathering_gap` of another (in their own box heights, ground points), standing still, kept for
           `gathering_hold_s` -> event "gathering" (start and end, edge_agent.events.Episode).
"""
from __future__ import annotations

import base64

import numpy as np

from .camera_profile import in_zone
from .events import Episode
from .slow_lane import next_event
from .slow_tasks import TrackMotion, Windows, stream_fields

PERSON = 0
DEFAULTS = {"crowd_persons": 30, "crowd_overlap_share": .3, "crowd_overlap_min_persons": 8, "crowd_every_s": 5.0,
            "crowd_long_side": 1280, "crowd_jpeg_quality": 85, "crowd_deadline_s": 10.0,
            "gathering_persons": 5, "gathering_gap": 1.5, "gathering_hold_s": 30.0, "window_s": 5.0,
            "stationary_speed": .3}


def overlap_share(boxes, iou_min: float = .3) -> float:
    """Share of person boxes overlapping at least one other by IoU >= iou_min."""
    def iou(a, b):
        w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
        h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
        inter = w * h
        union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
        return inter / union if union > 0 else 0.0
    if len(boxes) < 2:
        return 0.0
    hit = sum(any(iou(a, b) >= iou_min for j, b in enumerate(boxes) if j != i) for i, a in enumerate(boxes))
    return hit / len(boxes)


def encode_image(frame, ring, long_side: int, quality: int):
    """The full-resolution frame of this instant (own copy), else the newest the ring holds (kept 1 s: a crowd changes
    slowly), long side `long_side`, JPEG -> (base64, (w, h), frame used, full resolution?) or None. Only without any
    full-resolution frame (no branch on this camera) the analytics frame at the camera's aspect is used: fewer pixels,
    DM-Count undercounts far heads (640: MAE 13.3 vs 9.0). 10/11 on 246: "newest within 2 frames" missed 2 frames in 5
    (full resolution every 5th) and those fell back to 640: -6 to -16 people on vn6."""
    import cv2
    src = frame if getattr(frame, "hires", None) is not None else (ring.hires[-1] if ring.hires else None)
    if src is not None:
        img = src.hires
    else:
        from .events import snapshot_b64   # stretched analytics frame -> camera aspect
        b64 = snapshot_b64(frame, quality)
        if b64 is None:
            return None
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        src = frame
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    full = getattr(src, "hires", None) is not None
    return (base64.b64encode(jpeg.tobytes()).decode(), (img.shape[1], img.shape[0]), src, full) if ok else None


class CrowdTask:
    name = "crowd"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    async def run(self, queue, ring, stop) -> dict:
        cfg = self.cfg
        throttle = cfg.get("throttle") or (lambda: 1.0)
        area = (cfg.get("zones") or {}).get("area")
        next_ns, stats = 0, {"dense_frames": 0, "sent": 0, "no_image": 0, "sent_low_res": 0, "waited": 0}
        while (event := await next_event(queue, stop)) is not None:
            frame = event.frame
            people = [b for b in event.result.boxes if b.class_id == PERSON and in_zone(b, area)]
            share = overlap_share(people) if len(people) >= cfg["crowd_overlap_min_persons"] else 0.0
            if len(people) < cfg["crowd_persons"] and share < cfg["crowd_overlap_share"]:
                continue
            stats["dense_frames"] += 1
            t = frame.capture_mono_ns
            if t < next_ns:
                continue
            if getattr(frame, "hires", None) is None and ring.hires:
                # 15/11: wait for a dense frame with its own full copy, so frame_id, persons_boxes and the picture agree
                stats["waited"] += 1
                continue
            image = encode_image(frame, ring, int(cfg["crowd_long_side"]), int(cfg["crowd_jpeg_quality"]))
            if image is None:
                stats["no_image"] += 1
                continue
            b64, (w, h), src, full = image
            stats["sent_low_res"] += not full
            next_ns = t + round(cfg["crowd_every_s"] * throttle() * 1e9)
            capture = src.identity.capture_utc_ns
            stats["sent"] += 1
            if self.sink is not None:
                await self.sink("crowd", {**stream_fields(frame), "frame_id": src.identity.source_frame_id,
                                          "capture_utc_ns": capture,
                                          "deadline_utc_ns": capture + round(cfg["crowd_deadline_s"] * 1e9),
                                          "persons_boxes": len(people), "overlap_share": round(share, 3),
                                          "width": w, "height": h, "jpeg_b64": b64})
        return stats


class GatheringTask:
    name = "gathering"

    def __init__(self, sink=None, config=None) -> None:
        self.sink, self.cfg = sink, {**DEFAULTS, **(config or {})}

    @staticmethod
    def largest_group(people, gap: float) -> int:
        """Biggest set of people linked by ground-point distance <= gap x their mean box height (union-find)."""
        parent = list(range(len(people)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        pts = [((b.left + b.right) / 2, b.bottom, b.bottom - b.top) for b in people]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                (x1, y1, h1), (x2, y2, h2) = pts[i], pts[j]
                if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5 <= gap * (h1 + h2) / 2:
                    parent[root(i)] = root(j)
        sizes = {}
        for i in range(len(pts)):
            sizes[root(i)] = sizes.get(root(i), 0) + 1
        return max(sizes.values(), default=0)

    async def run(self, queue, ring, stop) -> dict:
        cfg = self.cfg
        watch = (cfg.get("zones") or {}).get("watch")
        hold = max(1, round(cfg["gathering_hold_s"] / cfg["window_s"]))
        episode, windows, motion = Episode("gathering", hold, hold), Windows(cfg["window_s"]), TrackMotion()
        best = 0
        while (event := await next_event(queue, stop)) is not None:
            frame = event.frame
            t = frame.capture_mono_ns
            still = []
            for b in event.result.boxes:
                if b.class_id != PERSON or b.track_id is None or not in_zone(b, watch):
                    continue
                speed = motion.update(b.track_id, t, b)
                if speed is not None and speed < cfg["stationary_speed"]:
                    still.append(b)
            best = max(best, self.largest_group(still, cfg["gathering_gap"]))
            if not windows.closes(t):
                continue
            message = episode.update(best >= cfg["gathering_persons"], frame, {"group_max": best})
            best = 0
            if message is not None and self.sink is not None:
                await self.sink("event", message)
        return {"events_started": episode.started, "open_at_end": episode.active is not None}
