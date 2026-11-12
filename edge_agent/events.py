"""Events from the slow lane (P1, 10/11): episodes, a durable outbox, snapshots, and the sink the tasks write to.

Episode: one rule's condition, checked once per window, becomes start/end events (start after `hold_on` true windows,
end after `hold_off` false ones), so a flickering condition never floods alerts. Every episode has one event_id; its
start message carries a snapshot (JPEG) of the frame.

EventOutbox: an event is appended to a local file BEFORE it is sent and acknowledged only after Kafka confirmed it, so
alerts survive a network or Cloud outage (sent again later; the database upserts by event_id, readers dedupe by
event_id + phase). Counts and plate crops are not kept: they are periodic and losing a few is harmless.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path


def snapshot_b64(frame, quality: int = 80) -> str | None:
    """JPEG of the analysed frame at the camera's aspect (hardware frames are stretched to 640x640)."""
    import cv2
    pixels = getattr(frame, "pixels", None)
    if pixels is None:
        return None
    size = getattr(frame, "source_size", None)
    if size is not None:
        w, h = size
        side = max(pixels.shape[:2])
        pixels = cv2.resize(pixels, (round(side * w / max(w, h)), round(side * h / max(w, h))),
                            interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(jpeg.tobytes()).decode() if ok else None


class Episode:
    """start/end of one condition with hysteresis; feed it once per window."""

    def __init__(self, event_type: str, hold_on: int, hold_off: int) -> None:
        if hold_on < 1 or hold_off < 1:
            raise ValueError("hold_on and hold_off are at least one window")
        self.event_type, self.hold_on, self.hold_off = event_type, hold_on, hold_off
        self.on = self.off = 0
        self.active = None   # the started event's fields while it lasts
        self.started = 0

    def update(self, condition: bool, frame, detail: dict, key: str = "") -> dict | None:
        """-> an event message to send (phase start or end), or None. key separates parallel episodes of one camera
        (e.g. one per loitering person)."""
        self.on, self.off = (self.on + 1, 0) if condition else (0, self.off + 1)
        i = frame.identity
        if self.active is None and self.on >= self.hold_on:
            self.started += 1
            self.active = {"node_id": i.node_id, "camera_id": i.camera_id, "stream_id": i.stream_id,
                           "stream_epoch": i.stream_epoch, "event_type": self.event_type,
                           "event_id": f"{i.node_id}/{i.camera_id}/{i.stream_id}/{i.stream_epoch}/{self.event_type}"
                                       f"{'/' + key if key else ''}/{i.capture_utc_ns}",
                           "started_utc_ns": i.capture_utc_ns}
            return {**self.active, "phase": "start", "detail": detail, "snapshot_jpeg_b64": snapshot_b64(frame)}
        if self.active is not None and self.off >= self.hold_off:
            event, self.active = {**self.active, "phase": "end", "ended_utc_ns": i.capture_utc_ns, "detail": detail}, None
            return event
        return None


class EventOutbox:
    """Append-only JSONL: {"add": key, "event": ...} then {"ack": key}. Pending = added and not acked."""

    def __init__(self, path: str | Path, max_pending: int = 10_000) -> None:
        self.path, self.max_pending = Path(path), max_pending
        self.pending: dict[str, dict] = {}
        self.dropped = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue   # a line cut by a crash
                if "add" in rec:
                    self.pending[rec["add"]] = rec["event"]
                elif "ack" in rec:
                    self.pending.pop(rec["ack"], None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._compact()

    @staticmethod
    def key(event: dict) -> str:
        return f"{event['event_id']}#{event['phase']}"

    def _write(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _compact(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for k, e in self.pending.items():
                f.write(json.dumps({"add": k, "event": e}, separators=(",", ":")) + "\n")
        os.replace(tmp, self.path)

    def add(self, event: dict) -> str:
        k = self.key(event)
        if len(self.pending) >= self.max_pending:   # shortcut: oldest dropped past 10k unsent alerts (hours offline)
            self.pending.pop(next(iter(self.pending)))
            self.dropped += 1
        self.pending[k] = event
        self._write({"add": k, "event": event})
        return k

    def ack(self, k: str) -> None:
        if self.pending.pop(k, None) is not None:
            self._write({"ack": k})
            if not self.pending:
                self._compact()   # keeps the file small: empty whenever nothing is owed


class SlowSink:
    """What slow tasks call: sink(kind, payload). Events go through the outbox and wait for Kafka's ack; the rest is
    sent without waiting. send(kind, payload, wait) is the transport (KafkaCloudClient.publish_slow)."""

    def __init__(self, send, outbox: EventOutbox | None = None, retry_s: float = 5.0) -> None:
        self.send, self.outbox, self.retry_s = send, outbox, retry_s
        self.stats = {"events_sent": 0, "events_failed": 0, "events_resent": 0, "other_sent": 0, "other_failed": 0}

    async def __call__(self, kind: str, payload: dict) -> None:
        if kind != "event":
            try:
                await self.send(kind, payload, False)
                self.stats["other_sent"] += 1
            except Exception:
                self.stats["other_failed"] += 1   # periodic data: the next window carries on
            return
        k = self.outbox.add(payload) if self.outbox is not None else None
        try:
            await self.send(kind, payload, True)
            self.stats["events_sent"] += 1
            if k is not None:
                self.outbox.ack(k)
        except Exception:
            self.stats["events_failed"] += 1   # stays in the outbox; resend() retries it

    async def resend(self) -> int:
        """Send what the outbox still owes (e.g. events of an earlier run or a past outage); -> how many went out."""
        if self.outbox is None:
            return 0
        sent = 0
        for k, event in list(self.outbox.pending.items()):
            try:
                await self.send("event", event, True)
            except Exception:
                break   # still offline: try again later
            self.outbox.ack(k)
            sent += 1
        self.stats["events_resent"] += sent
        return sent

    async def run_retries(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.resend()
            try:
                await asyncio.wait_for(stop.wait(), self.retry_s)
            except asyncio.TimeoutError:
                pass
        await self.resend()
