"""UC3 on the Cloud (P2, 10/11): dense-scene frames -> DM-Count on Triton -> people estimate + where they are.

Registered in cloud_core/slow_worker.py (import = registration). Per frame: one row on cloud.crowd.v1 (table crowd:
estimate, the fast lane's person-box count, an 8 x 8 grid of people per cell). An alert episode per camera: "crowd"
starts after UC3_ALERT_HOLD frames with an estimate >= UC3_ALERT_PERSONS and ends after as many below it, or when the
Edge stopped sending (the scene is no longer dense) for UC3_ALERT_SILENCE_S.
"""
from __future__ import annotations

import asyncio
import base64
import os
from time import monotonic, perf_counter, time_ns
from types import SimpleNamespace

import numpy as np

from cloud_core.slow_worker import handler, sweeper
from edge_agent.events import Episode
from inference.crowd_density import prepare_square, summarize

IN_TOPIC, OUT_TOPIC, EVENTS_TOPIC = "edge.crowd-frame.v1", "cloud.crowd.v1", "edge.event.v1"
MODEL = os.getenv("UC3_TRITON_MODEL", "crowd_density")
ALERT = float(os.getenv("UC3_ALERT_PERSONS", "100"))
HOLD = int(os.getenv("UC3_ALERT_HOLD", "3"))
SILENCE_S = float(os.getenv("UC3_ALERT_SILENCE_S", "60"))
KEYS = ("node_id", "camera_id", "stream_id", "stream_epoch")

_session = None
_episodes: dict[tuple, list] = {}   # stream -> [Episode, last frame stand-in, monotonic time of its last frame]


def run_model(x):
    """1x3x1280x1280 -> density map. Triton (batched with every other camera's frames, low priority)."""
    global _session
    if _session is None:
        from inference.triton_detr import TritonSession
        _session = TritonSession(MODEL)
    return _session.run(["density"], {"image": x})[0]


def stand_in(m: dict):
    """What Episode needs of a frame: its identity (no pixels: the Cloud attaches the received JPEG itself)."""
    ident = SimpleNamespace(**{k: m[k] for k in KEYS}, capture_utc_ns=m["capture_utc_ns"])
    return SimpleNamespace(identity=ident, pixels=None)


def _event(message, jpeg_b64=None) -> dict:
    out = {"schema_version": 1, "kind": "event", **message}
    if message["phase"] == "start":
        out["snapshot_jpeg_b64"] = jpeg_b64
    return out


@handler(IN_TOPIC)
async def crowd_frame(m: dict, model=None) -> list:
    import cv2
    model = model or run_model
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(m["jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("crowd frame is not a JPEG")
    x, valid = prepare_square(bgr)
    began = perf_counter()
    density = await asyncio.to_thread(model, x)
    infer_ms = (perf_counter() - began) * 1000
    s = summarize(density, valid)
    key = "/".join(str(m[k]) for k in KEYS).encode()
    row = {"schema_version": 1, "kind": "crowd", **{k: m[k] for k in KEYS}, "capture_utc_ns": m["capture_utc_ns"],
           "frame_id": m["frame_id"], "persons_boxes": m.get("persons_boxes"), "persons_est": s["count"],
           "grid": s["grid"], "infer_ms": round(infer_ms, 1)}
    out = [(OUT_TOPIC, key, row)]
    stream = tuple(m[k] for k in KEYS)
    state = _episodes.setdefault(stream, [Episode("crowd", HOLD, HOLD), None, 0.0])
    frame = stand_in(m)
    state[1], state[2] = frame, monotonic()
    top = sorted(((v, r, c) for r, line in enumerate(s["grid"]) for c, v in enumerate(line)), reverse=True)[:3]
    message = state[0].update(s["count"] >= ALERT, frame, {"persons_est": s["count"],
                                                           "dense_cells": [[r, c, v] for v, r, c in top]})
    if message is not None:
        out.append((EVENTS_TOPIC, key, _event(message, m["jpeg_b64"])))
    return out


@sweeper
async def close_silent(now: float | None = None) -> list:
    """An open crowd alert whose Edge stopped sending frames (no longer dense) for SILENCE_S ends there."""
    now = monotonic() if now is None else now
    out = []
    for stream, (episode, frame, last) in list(_episodes.items()):
        if episode.active is not None and now - last > SILENCE_S:
            ended = {**episode.active, "phase": "end", "ended_utc_ns": time_ns(),
                     "detail": {"reason": "no dense frames"}}
            episode.active, episode.on, episode.off = None, 0, 0
            out.append((EVENTS_TOPIC, "/".join(map(str, stream)).encode(), _event(ended)))
        if episode.active is None and now - last > SILENCE_S:
            del _episodes[stream]   # bounded: streams that went quiet are forgotten
    return out
