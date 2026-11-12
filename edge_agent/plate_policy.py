"""Which plate crops an Edge sends to the Cloud OCR (pure logic, no model).

A plate track is read as soon as it appears, then again every `every` frames while it stays in view, until `confirm`
valid VN reads agree or `cap` reads were sent (0 = no limit). Tracks match by IoU as in the team's plate_reader.py.
`every=3, cap=0, confirm=0` is that repo's main.py; `cap=1` is one read per plate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from inference.plate_ocr import valid_plate


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PlateTrack:
    track_id: int
    box: tuple
    first_frame: int
    last_frame: int
    seen: int = 1
    reads: int = 0
    last_read: int | None = None
    votes: Counter = field(default_factory=Counter)  # valid texts only
    texts: list = field(default_factory=list)  # every reply, in order
    end: str | None = None  # confirmed | cap | lost

    def best(self) -> str:
        if self.votes:
            return self.votes.most_common(1)[0][0]
        return Counter(t for t in self.texts if t).most_common(1)[0][0] if any(self.texts) else ""

    def summary(self) -> dict:
        return {"event": "track", "track": self.track_id, "first_frame": self.first_frame,
                "last_frame": self.last_frame, "frames_seen": self.seen, "reads": self.reads,
                "replies": len(self.texts), "valid_replies": sum(self.votes.values()), "text": self.best(),
                "end": self.end or "lost"}


class PlateReadPolicy:
    def __init__(self, *, every: int = 3, cap: int = 5, confirm: int = 2, max_age: int = 15,
                 iou_thr: float = .3, min_px: int = 20) -> None:
        if every < 1 or cap < 0 or confirm < 0 or max_age < 0 or not 0 < iou_thr < 1:
            raise ValueError("every >= 1, cap/confirm/max_age >= 0, 0 < iou_thr < 1")
        self.every, self.cap, self.confirm, self.max_age = every, cap, confirm, max_age
        self.iou_thr, self.min_px = iou_thr, min_px
        self.tracks: list[PlateTrack] = []
        self.finished: list[dict] = []
        self._next_id = 0

    def update(self, frame: int, boxes) -> list[PlateTrack]:
        """boxes: (x1, y1, x2, y2[, conf]) in frame pixels, one call per analysed frame -> tracks to read now."""
        alive = []
        for t in self.tracks:
            if frame - t.last_frame > self.max_age:
                self.finished.append(t.summary())
            else:
                alive.append(t)
        self.tracks = alive
        free, read = list(self.tracks), []
        for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
            box = tuple(box[:4])
            best, best_iou = None, self.iou_thr
            for t in free:  # one box per track per frame
                v = iou(t.box, box)
                if v > best_iou:
                    best, best_iou = t, v
            if best is None:
                best = PlateTrack(self._next_id, box, frame, frame, seen=0)
                self._next_id += 1
                self.tracks.append(best)
            else:
                free.remove(best)
            best.box, best.last_frame = box, frame
            best.seen += 1
            big = min(box[2] - box[0], box[3] - box[1]) >= self.min_px
            due = best.last_read is None or frame - best.last_read >= self.every
            if best.end is None and big and due:
                best.reads += 1
                best.last_read = frame
                if self.cap and best.reads >= self.cap:
                    best.end = "cap"
                read.append(best)
        return read

    def cancel(self, track: PlateTrack) -> None:
        """The crop picked by update() was not sent (Edge back-pressure): it does not count as a read."""
        track.reads -= 1
        if track.end == "cap":
            track.end = None

    def record(self, track: PlateTrack, text: str) -> None:
        track.texts.append(text)
        if valid_plate(text):
            track.votes[text] += 1
            if self.confirm and track.end in (None, "cap") and track.votes[text] >= self.confirm:
                track.end = "confirmed"

    def close(self) -> list[dict]:
        self.finished += [t.summary() for t in self.tracks]
        self.tracks = []
        return self.finished
