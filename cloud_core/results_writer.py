"""Results -> database (v4, 6/11): reads the Cloud result topics and writes batches, outside the detection path.

A slow or dead database never slows detection: this process only reads Kafka. While the database is down it keeps
retrying with the rows it holds, up to DB_MAX_PENDING_ROWS, then drops them (counted in the log).

  DB_DSN=postgresql://... KAFKA_BOOTSTRAP_SERVERS=localhost:29092 python -m cloud_core.results_writer
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from time import monotonic

from cloud_core.media_store import MEDIA_DIR, prune, save_media
from contracts.grpc_codec import result_from_proto
from contracts.proto import edge_video_platform_pb2 as pb

TOPICS = ("controlv1.cloud.detection-result.v1", "cloud.detection-result.v1", "edge.tracked-result.v1",  # mode A
          "edge.count.v1", "edge.event.v1", "cloud.plate-read.v1", "cloud.crowd.v1",  # slow lane (8-10/11)
          "cloud.action.v1")  # UC4 check scores (12/11)
SLOW_TOPICS = ("edge.count.v1", "edge.event.v1", "cloud.plate-read.v1", "cloud.crowd.v1", "cloud.action.v1")
EVENTS_KEEP_DAYS = float(os.getenv("V4_EVENTS_KEEP_DAYS", "90"))   # events had no retention (12/11)
SKIPPED_DECISIONS = ("probe/", "load/", "benchmark/")  # measurements, not traffic results
FRAME_COLUMNS = ("capture_time", "node_id", "camera_id", "stream_id", "stream_epoch", "frame_id", "media_pts_ns",
                 "model_id", "model_mode", "box_count", "latency_ms")
BOX_COLUMNS = ("capture_time", "node_id", "camera_id", "stream_id", "stream_epoch", "frame_id", "media_pts_ns",
               "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "model_mode")


def rows_from_message(value: bytes):
    """Kafka result envelope -> (frame row, box rows), or None when it is not a stored result."""
    envelope = json.loads(value)
    if envelope.get("status") != "accepted" or not envelope.get("result"):
        return None
    result = result_from_proto(pb.DetectionResult.FromString(bytes.fromhex(envelope["result"])))
    if result.decision_id.startswith(SKIPPED_DECISIONS):
        return None
    f = result.frame
    when = datetime.fromtimestamp(f.capture_utc_ns / 1e9, tz=timezone.utc)
    head = (when, f.node_id, f.camera_id, f.stream_id, f.stream_epoch, f.source_frame_id, result.preview_pts)
    mode = result.model_mode.value
    frame = head + (result.model_id, mode, len(result.boxes), (result.produced_utc_ns - f.capture_utc_ns) / 1e6)
    boxes = [head + (b.track_id, b.class_id, b.class_name, b.confidence, b.left, b.top, b.right, b.bottom, mode)
             for b in result.boxes]
    return frame, boxes


COUNT_COLUMNS = ("window_end", "node_id", "camera_id", "stream_id", "stream_epoch", "window_s", "frames", "persons",
                 "bicycles", "cars", "motorcycles", "buses", "trucks", "vehicles_max", "persons_max", "new_tracks",
                 "stationary_share")
EVENT_UPSERT = """INSERT INTO events (event_id, event_type, node_id, camera_id, stream_id, stream_epoch, started_at,
    ended_at, detail, snapshot_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (event_id) DO UPDATE SET ended_at = COALESCE(EXCLUDED.ended_at, events.ended_at),
    detail = events.detail || EXCLUDED.detail, snapshot_path = COALESCE(events.snapshot_path, EXCLUDED.snapshot_path)"""



PLATE_COLUMNS = ("capture_time", "node_id", "camera_id", "stream_id", "stream_epoch", "track_id", "frame_id",
                 "vehicle_class", "read_no", "text", "valid", "det_conf", "ocr_ms", "image_path")


CROWD_COLUMNS = ("capture_time", "node_id", "camera_id", "stream_id", "stream_epoch", "frame_id", "persons_boxes",
                 "persons_est", "grid", "infer_ms")


# UC4 (12/11): one row per Cloud check (cloud_core/uc4_disorder): fight clip score (P punch + kick + push) or crash
# (someone lying); detail = the rest (frames with people, persons lying...). For setting UC4_FIGHT_SCORE after retraining.
ACTION_COLUMNS = ("capture_time", "node_id", "camera_id", "stream_id", "stream_epoch", "frame_id", "check_type", "score",
                  "detail", "infer_ms")


def _utc(ns):
    return None if ns is None else datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def slow_row(value: bytes, media_dir: str = MEDIA_DIR):
    """edge.count.v1 / edge.event.v1 message -> ("count", row) or ("event", upsert params)."""
    m = json.loads(value)
    if m.get("kind") == "count":
        m = {**m, "window_end": _utc(m["window_end_utc_ns"])}
        return "count", tuple(m[c] for c in COUNT_COLUMNS)
    if m.get("kind") == "action":   # cloud.action.v1 (cloud_core/uc4_disorder)
        keys = set(ACTION_COLUMNS) | {"kind", "schema_version", "check", "capture_utc_ns"}
        m = {**m, "capture_time": _utc(m["capture_utc_ns"]), "check_type": m["check"], "infer_ms": m.get("infer_ms"),
             "detail": json.dumps({k: v for k, v in m.items() if k not in keys})}
        return "action", tuple(m[c] for c in ACTION_COLUMNS)
    if m.get("kind") == "crowd":   # cloud.crowd.v1 (cloud_core/uc3_crowd)
        m = {**m, "capture_time": _utc(m["capture_utc_ns"]), "grid": json.dumps(m["grid"])}
        return "crowd", tuple(m[c] for c in CROWD_COLUMNS)
    if "text" in m and "track_id" in m:   # cloud.plate-read.v1 (cloud_core/plate_worker)
        m = {"image_path": None, **m, "capture_time": _utc(m["capture_utc_ns"])}
        return "plate", tuple(m[c] for c in PLATE_COLUMNS)
    if m.get("kind") == "event":
        detail = {f"{m['phase']}": m.get("detail", {})}   # {"start": {...}} then merged with {"end": {...}}
        snapshot = save_media("events", f"{m['event_id']}_{m['phase']}", m.get("snapshot_jpeg_b64"), media_dir)
        return "event", (m["event_id"], m["event_type"], m["node_id"], m["camera_id"], m["stream_id"],
                         m["stream_epoch"], _utc(m["started_utc_ns"]), _utc(m.get("ended_utc_ns")),
                         json.dumps(detail), snapshot)
    raise ValueError(f"unknown slow lane message kind {m.get('kind')!r}")


async def write_batch(conn, frames, boxes, counts=(), events=(), plates=(), crowd=(), actions=()):
    async with conn.transaction():
        async with conn.cursor() as cur:
            async with cur.copy(f"COPY frames ({','.join(FRAME_COLUMNS)}) FROM STDIN") as copy:
                for row in frames:
                    await copy.write_row(row)
            async with cur.copy(f"COPY detections ({','.join(BOX_COLUMNS)}) FROM STDIN") as copy:
                for row in boxes:
                    await copy.write_row(row)
            if counts:
                async with cur.copy(f"COPY counts ({','.join(COUNT_COLUMNS)}) FROM STDIN") as copy:
                    for row in counts:
                        await copy.write_row(row)
            if crowd:
                async with cur.copy(f"COPY crowd ({','.join(CROWD_COLUMNS)}) FROM STDIN") as copy:
                    for row in crowd:
                        await copy.write_row(row)
            if plates:
                async with cur.copy(f"COPY plates ({','.join(PLATE_COLUMNS)}) FROM STDIN") as copy:
                    for row in plates:
                        await copy.write_row(row)
            if actions:
                async with cur.copy(f"COPY actions ({','.join(ACTION_COLUMNS)}) FROM STDIN") as copy:
                    for row in actions:
                        await copy.write_row(row)
            for params in events:   # few; start then end of one episode stay in order
                await cur.execute(EVENT_UPSERT, params)


async def main():
    import psycopg
    from aiokafka import AIOKafkaConsumer
    dsn, period = os.environ["DB_DSN"], float(os.getenv("DB_FLUSH_S", "0.5"))
    limit = int(os.getenv("DB_MAX_PENDING_ROWS", "200000"))  # bounded memory while the database is down
    consumer = AIOKafkaConsumer(*TOPICS, bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"],
                                group_id="v4-results-writer", auto_offset_reset="latest", enable_auto_commit=True)
    await consumer.start()
    print(f"[results_writer] reading {TOPICS}", flush=True)
    conn, frames, boxes, counts, events, plates, crowd, actions = None, [], [], [], [], [], [], []
    written, dropped, bad = 0, 0, 0
    last, log_at, pruned_at = monotonic(), monotonic(), 0.0
    try:
        while True:
            batch = await consumer.getmany(timeout_ms=int(period * 500), max_records=2000)
            for messages in batch.values():
                for message in messages:
                    try:
                        if message.topic in SLOW_TOPICS:
                            kind, row = slow_row(message.value)
                            {"count": counts, "event": events, "plate": plates, "crowd": crowd,
                             "action": actions}[kind].append(row)
                            continue
                        rows = rows_from_message(message.value)
                    except Exception as exc:  # one malformed message must not stop the writer
                        bad += 1
                        print(f"[results_writer] bad message: {exc}", flush=True)
                        continue
                    if rows is not None:
                        frames.append(rows[0])
                        boxes.extend(rows[1])
            if (frames or counts or events or plates or crowd or actions) and monotonic() - last >= period:
                last = monotonic()
                try:
                    if conn is None or conn.closed:
                        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
                    await write_batch(conn, frames, boxes, counts, events, plates, crowd, actions)
                    written += len(frames)
                    frames, boxes, counts, events, plates, crowd, actions = [], [], [], [], [], [], []
                except psycopg.Error as exc:
                    print(f"[results_writer] database error, will retry: {exc}", flush=True)
                    conn = None
                    if len(boxes) + len(frames) > limit:
                        dropped += len(frames)
                        frames, boxes, counts, crowd, actions = [], [], [], [], []   # events kept: few, the alerts
            if monotonic() - pruned_at >= 3600:   # hourly: evidence files and old events (no retention on either)
                pruned_at = monotonic()
                try:
                    files = await asyncio.to_thread(prune)
                    if conn is not None and not conn.closed:
                        await conn.execute("DELETE FROM events WHERE started_at < now() - make_interval(days => %s)",
                                           (int(EVENTS_KEEP_DAYS),))
                    if files:
                        print(f"[results_writer] pruned {files} evidence files", flush=True)
                except Exception as exc:   # housekeeping never stops the writer
                    print(f"[results_writer] prune failed: {exc}", flush=True)
            if monotonic() - log_at >= 30:
                log_at = monotonic()
                print(f"[results_writer] frames written {written}, dropped {dropped}, bad {bad}, "
                      f"pending {len(frames)}", flush=True)
    finally:
        await consumer.stop()
        if conn is not None:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
