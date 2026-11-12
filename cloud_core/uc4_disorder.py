"""UC4 on the Cloud (P3, 11/11): checks what the Edge rules flagged (edge_agent/disorder_task.py).

edge.action-clip.v1  a ~2 s clip of crops around people fighting-or-not -> RTMO-s on every crop (Triton pose_rtmo_s,
                     batched) -> the 2 people closest to the crop centre, followed frame to frame -> ST-GCN++ (Triton
                     action_stgcnpp) -> P(punch + kick + push) >= UC4_FIGHT_SCORE: event "fight".
edge.crash-frame.v1  one crop around a fallen bike -> RTMO-s -> someone lying (torso horizontal) -> event "crash".
Each check also publishes a row on cloud.action.v1 (score, people seen) - for tuning thresholds; not stored yet.
An event per camera and type: starts with the first confirmed message, ends after UC4_SILENCE_S without one.
Registered in cloud_core/slow_worker.py (import = registration).
"""
from __future__ import annotations

import asyncio
import base64
import os
from time import monotonic, perf_counter, time_ns

import numpy as np

from cloud_core.slow_worker import handler, sweeper
from cloud_core.uc3_crowd import KEYS, _event, stand_in
from edge_agent.events import Episode
from inference.action_skeleton import fight_score, prepare_clip
from inference.pose_rtmo import letterbox, lying, people

CLIPS_TOPIC, CRASH_TOPIC = "edge.action-clip.v1", "edge.crash-frame.v1"
OUT_TOPIC, EVENTS_TOPIC = "cloud.action.v1", "edge.event.v1"
POSE_MODEL = os.getenv("UC4_POSE_MODEL", "pose_rtmo_s")
ACTION_MODEL = os.getenv("UC4_ACTION_MODEL", "action_stgcnpp")
FIGHT_SCORE = float(os.getenv("UC4_FIGHT_SCORE", "0.3"))   # NTU60 weights: to set again after retraining
SILENCE_S = float(os.getenv("UC4_SILENCE_S", "30"))
POSE_BATCH = 4   # the Triton config's max batch

_sessions: dict[str, object] = {}
_episodes: dict[tuple, list] = {}   # (stream..., event type) -> [Episode, monotonic time of the last confirmation]


def _session(model: str):
    if model not in _sessions:
        from inference.triton_detr import TritonSession
        _sessions[model] = TritonSession(model)
    return _sessions[model]


def run_pose(x):
    """Bx3x640x640 -> (dets, keypoints)."""
    return _session(POSE_MODEL).run(["dets", "keypoints"], {"input": x})


def run_action(x):
    """1x2x100x17x3 -> softmax 1x60."""
    return _session(ACTION_MODEL).run(["prob"], {"skeleton": x})[0]


def decode(b64: str):
    import cv2
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("UC4 crop is not a JPEG")
    return bgr


async def poses(images, pose=None) -> list:
    """Every image's people [(score, box, kp 17x3)], RTMO-s in batches of POSE_BATCH."""
    pose = pose or run_pose
    out = []
    for i in range(0, len(images), POSE_BATCH):
        xs, rs = zip(*(letterbox(img) for img in images[i:i + POSE_BATCH]))
        dets, kps = await asyncio.to_thread(pose, np.concatenate(xs))
        out += [people(dets[k], kps[k], rs[k]) for k in range(len(rs))]
    return out


def follow(frames_people, center, count: int = 2) -> list:
    """Per frame the keypoints (people, 17, 3) of `count` people: the ones closest to `center` where they first
    appear, then each frame the person nearest each one's last position (no tracker on the Cloud; a 2 s clip)."""
    def mid(p):
        return (p[1][:2] + p[1][2:]) / 2
    last, seq = None, []
    for found in frames_people:
        if last is None and found:
            last = [mid(p) for p in sorted(found, key=lambda p: float(np.linalg.norm(mid(p) - center)))[:count]]
        if last is None or not found:
            seq.append(np.zeros((0, 17, 3), np.float32))
            continue
        picked, free = [], list(range(len(found)))
        for k, at in enumerate(last):
            if not free:
                break
            j = min(free, key=lambda q: float(np.linalg.norm(mid(found[q]) - at)))
            free.remove(j)
            picked.append(found[j][2])
            last[k] = mid(found[j])
        seq.append(np.stack(picked).astype(np.float32))
    return seq


def _confirm(m: dict, event_type: str, confirmed: bool, detail: dict, jpeg_b64: str, key: bytes) -> list:
    stream = tuple(m[k] for k in KEYS) + (event_type,)
    if not confirmed:
        return []
    state = _episodes.setdefault(stream, [Episode(event_type, 1, 1), 0.0])
    state[1] = monotonic()
    message = state[0].update(True, stand_in(m), detail)
    return [] if message is None else [(EVENTS_TOPIC, key, _event(message, jpeg_b64))]


@handler(CLIPS_TOPIC)
async def action_clip(m: dict, pose=None, action=None) -> list:
    images = [decode(f["jpeg_b64"]) for f in m["frames"]]
    began = perf_counter()
    found = await poses(images, pose)
    h, w = images[0].shape[:2]
    seq = follow(found, np.array([w / 2, h / 2]))
    seen = sum(len(s) > 0 for s in seq)
    score = 0.0
    if seen >= 2:
        prob = await asyncio.to_thread(action or run_action, prepare_clip(seq, (h, w)))
        score = fight_score(prob)
    key = "/".join(str(m[k]) for k in KEYS).encode()
    row = {"schema_version": 1, "kind": "action", **{k: m[k] for k in KEYS}, "capture_utc_ns": m["capture_utc_ns"],
           "frame_id": m["frame_id"], "check": "fight", "score": round(score, 4), "frames": len(images),
           "frames_with_people": seen, "infer_ms": round((perf_counter() - began) * 1000, 1)}
    middle = m["frames"][len(m["frames"]) // 2]["jpeg_b64"]
    return [(OUT_TOPIC, key, row)] + _confirm(m, "fight", score >= FIGHT_SCORE,
                                              {"score": round(score, 3), "persons": m.get("persons"),
                                               "region": m.get("region")}, middle, key)


@handler(CRASH_TOPIC)
async def crash_frame(m: dict, pose=None) -> list:
    img = decode(m["jpeg_b64"])
    began = perf_counter()
    found = (await poses([img], pose))[0]
    down = [p for p in found if lying(p[2])]
    key = "/".join(str(m[k]) for k in KEYS).encode()
    row = {"schema_version": 1, "kind": "action", **{k: m[k] for k in KEYS}, "capture_utc_ns": m["capture_utc_ns"],
           "frame_id": m["frame_id"], "check": "crash", "score": float(len(down) > 0), "persons": len(found),
           "persons_lying": len(down), "infer_ms": round((perf_counter() - began) * 1000, 1)}
    return [(OUT_TOPIC, key, row)] + _confirm(m, "crash", bool(down),
                                              {"track_id": m.get("track_id"), "persons_lying": len(down),
                                               "box": m.get("box")}, m["jpeg_b64"], key)


@sweeper
async def close_quiet(now: float | None = None) -> list:
    """A fight / crash event with no new confirmation for SILENCE_S ends there."""
    now = monotonic() if now is None else now
    out = []
    for stream, (episode, last) in list(_episodes.items()):
        if now - last <= SILENCE_S:
            continue
        if episode.active is not None:
            ended = {**episode.active, "phase": "end", "ended_utc_ns": time_ns(), "detail": {"reason": "quiet"}}
            out.append((EVENTS_TOPIC, "/".join(map(str, stream[:-1])).encode(), _event(ended)))
        del _episodes[stream]
    return out
