-- v4 results store (6/11): what the Cloud produced, for the web team. PostgreSQL + TimescaleDB on 249.
-- frames: one row per analysed frame, also when it has no box (so a reader can clear old boxes).
-- detections: one row per box. Coordinates are 0..1 of the source frame; media_pts_ns matches the video PTS.
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS frames (
    capture_time  timestamptz NOT NULL,
    node_id       text NOT NULL,
    camera_id     text NOT NULL,
    stream_id     text NOT NULL,
    stream_epoch  bigint NOT NULL,
    frame_id      bigint NOT NULL,
    media_pts_ns  bigint,
    model_id      text NOT NULL,
    model_mode    text NOT NULL,
    box_count     integer NOT NULL,
    latency_ms    real            -- capture -> result produced on the Cloud
);
CREATE TABLE IF NOT EXISTS detections (
    capture_time  timestamptz NOT NULL,
    node_id       text NOT NULL,
    camera_id     text NOT NULL,
    stream_id     text NOT NULL,
    stream_epoch  bigint NOT NULL,
    frame_id      bigint NOT NULL,
    media_pts_ns  bigint,
    track_id      integer,
    class_id      smallint NOT NULL,
    class_name    text NOT NULL,
    confidence    real NOT NULL,
    x1 real NOT NULL, y1 real NOT NULL, x2 real NOT NULL, y2 real NOT NULL,
    model_mode    text NOT NULL
);
SELECT create_hypertable('frames', 'capture_time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
SELECT create_hypertable('detections', 'capture_time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS frames_stream_time ON frames (camera_id, stream_id, capture_time DESC);
CREATE INDEX IF NOT EXISTS detections_stream_frame ON detections (camera_id, stream_id, frame_id, capture_time DESC);
CREATE INDEX IF NOT EXISTS detections_track ON detections (camera_id, track_id, capture_time DESC);
SELECT add_retention_policy('frames', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('detections', INTERVAL '7 days', if_not_exists => TRUE);

-- slow lane (8/11). counts: one row per camera per window (default 5 s), UC3 counting from the boxes. People/vehicle
-- columns are means per analysed frame in the window; *_max the busiest frame; new_tracks = track IDs first seen.
CREATE TABLE IF NOT EXISTS counts (
    window_end       timestamptz NOT NULL,
    node_id          text NOT NULL,
    camera_id        text NOT NULL,
    stream_id        text NOT NULL,
    stream_epoch     bigint NOT NULL,
    window_s         real NOT NULL,
    frames           integer NOT NULL,
    persons          real NOT NULL,
    bicycles         real NOT NULL,
    cars             real NOT NULL,
    motorcycles      real NOT NULL,
    buses            real NOT NULL,
    trucks           real NOT NULL,
    vehicles_max     integer NOT NULL,
    persons_max      integer NOT NULL,
    new_tracks       integer NOT NULL,
    stationary_share real            -- share of tracked vehicles standing; NULL = none measurable
);
SELECT create_hypertable('counts', 'window_end', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS counts_stream_time ON counts (camera_id, stream_id, window_end DESC);
SELECT add_retention_policy('counts', INTERVAL '30 days', if_not_exists => TRUE);
-- events: one row per episode (UC2 congestion, later gathering, loitering, fall ...); ended_at NULL while it lasts.
CREATE TABLE IF NOT EXISTS events (
    event_id      text PRIMARY KEY,
    event_type    text NOT NULL,
    node_id       text NOT NULL,
    camera_id     text NOT NULL,
    stream_id     text NOT NULL,
    stream_epoch  bigint NOT NULL,
    started_at    timestamptz NOT NULL,
    ended_at      timestamptz,
    detail        jsonb NOT NULL DEFAULT '{}',
    snapshot_path text           -- JPEG of the frame when it started (P1, file on 249)
);
CREATE INDEX IF NOT EXISTS events_camera_time ON events (camera_id, started_at DESC);
CREATE INDEX IF NOT EXISTS events_open ON events (event_type) WHERE ended_at IS NULL;
ALTER TABLE events ADD COLUMN IF NOT EXISTS snapshot_path text;   -- databases created before P1

-- UC1 (8/11): one row per plate crop read by the Cloud OCR (at most a few per vehicle track); `valid` = VN plate format.
CREATE TABLE IF NOT EXISTS plates (
    capture_time   timestamptz NOT NULL,
    node_id        text NOT NULL,
    camera_id      text NOT NULL,
    stream_id      text NOT NULL,
    stream_epoch   bigint NOT NULL,
    track_id       integer NOT NULL,
    frame_id       bigint NOT NULL,
    vehicle_class  smallint NOT NULL,
    read_no        smallint NOT NULL,
    text           text NOT NULL,
    valid          boolean NOT NULL,
    det_conf       real,
    ocr_ms         real,
    image_path     text            -- the plate crop the OCR read (P1, file on 249)
);
SELECT create_hypertable('plates', 'capture_time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS plates_track ON plates (camera_id, stream_id, stream_epoch, track_id);
CREATE INDEX IF NOT EXISTS plates_text ON plates (text) WHERE valid;
SELECT add_retention_policy('plates', INTERVAL '30 days', if_not_exists => TRUE);
ALTER TABLE plates ADD COLUMN IF NOT EXISTS image_path text;   -- databases created before P1

-- UC3 dense scenes (P2, 10/11): one row per frame the Edge sent (dense: >= 30 people or overlapping boxes), every
-- 5 s or slower. persons_est = DM-Count estimate; persons_boxes = the fast lane's person boxes in the same zone;
-- grid = 8 x 8 people per cell over the image, row-major from the top left (where the crowd is).
CREATE TABLE IF NOT EXISTS crowd (
    capture_time   timestamptz NOT NULL,
    node_id        text NOT NULL,
    camera_id      text NOT NULL,
    stream_id      text NOT NULL,
    stream_epoch   bigint NOT NULL,
    frame_id       bigint NOT NULL,
    persons_boxes  integer,
    persons_est    real NOT NULL,
    grid           jsonb NOT NULL,
    infer_ms       real
);
SELECT create_hypertable('crowd', 'capture_time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS crowd_stream_time ON crowd (camera_id, stream_id, capture_time DESC);
SELECT add_retention_policy('crowd', INTERVAL '30 days', if_not_exists => TRUE);
-- UC4 (12/11): one row per Cloud check (cloud_core/uc4_disorder.py): check_type fight (score = P punch + kick + push from
-- ST-GCN++) or crash (score 1 = someone lying near the fallen bike); detail = frames with people, persons lying...
-- Used to set UC4_FIGHT_SCORE after retraining. The confirmed fights / crashes are rows of events.
CREATE TABLE IF NOT EXISTS actions (
    capture_time   timestamptz NOT NULL,
    node_id        text NOT NULL,
    camera_id      text NOT NULL,
    stream_id      text NOT NULL,
    stream_epoch   bigint NOT NULL,
    frame_id       bigint NOT NULL,
    check_type     text NOT NULL,
    score          real NOT NULL,
    detail         jsonb NOT NULL DEFAULT '{}',
    infer_ms       real
);
SELECT create_hypertable('actions', 'capture_time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS actions_stream_time ON actions (camera_id, stream_id, capture_time DESC);
SELECT add_retention_policy('actions', INTERVAL '30 days', if_not_exists => TRUE);
-- the plate of each vehicle track: its most frequent valid read (else the most frequent read)
CREATE OR REPLACE VIEW plate_tracks AS
WITH c AS (
    SELECT node_id, camera_id, stream_id, stream_epoch, track_id, text, bool_or(valid) AS valid, count(*) AS n,
           min(capture_time) AS first_seen, max(capture_time) AS last_seen
    FROM plates GROUP BY node_id, camera_id, stream_id, stream_epoch, track_id, text)
SELECT DISTINCT ON (node_id, camera_id, stream_id, stream_epoch, track_id)
       node_id, camera_id, stream_id, stream_epoch, track_id, text AS plate, valid, n AS reads_with_this_text,
       sum(n) OVER t AS reads, min(first_seen) OVER t AS first_seen, max(last_seen) OVER t AS last_seen
FROM c
WINDOW t AS (PARTITION BY node_id, camera_id, stream_id, stream_epoch, track_id)
ORDER BY node_id, camera_id, stream_id, stream_epoch, track_id, valid DESC, n DESC;
