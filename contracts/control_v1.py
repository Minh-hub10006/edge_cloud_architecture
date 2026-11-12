"""Additive control-v1 wire helpers; existing protobuf/RPCs stay unchanged."""
from __future__ import annotations

import json
import math
from typing import Any

SERVICE = "edge_video_platform.control.v1.DataPlane"
KAFKA_NAMESPACE = "controlv1"
FEEDBACK_TOPIC = "controlv1.cloud.feedback.v1"
VERSION = 1


def finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def encode(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def decode(value: bytes) -> Any:
    if len(value) > 65536:
        raise ValueError("control feedback exceeds 64 KiB")
    return json.loads(value)


def validate_feedback(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("control_v1") != VERSION:
        raise ValueError("invalid control feedback version")
    if not isinstance(value.get("cloud_instance_id"), str) or not value["cloud_instance_id"]:
        raise ValueError("missing Cloud instance identity")
    for name in ("sequence", "unknown_capacity_tickets"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError(f"invalid {name}")
    for name in ("window_ms", "measurement_to_send_ms"):
        finite(value.get(name), name, positive=name == "window_ms")
    for name in ("cloud_inflight_norm", "cloud_arrival_pressure_norm"):
        if value.get(name) is not None and not 0 <= finite(value[name], name) <= 1:
            raise ValueError(f"{name} outside [0,1]")
    for name in ("cloud_arrival_pressure_raw", "arrival_rate_rps"):
        if value.get(name) is not None:
            finite(value[name], name)
    if value.get("command") is not None:
        validate_command(value["command"])
    return dict(value)


def validate_command(command: dict) -> dict:
    """Per-stream Cloud scheduler command (docs/PLAN_CONTROL_V3_VI.md); the Edge still checks its allowed modes."""
    if not isinstance(command, dict):
        raise ValueError("command must be an object")
    if not isinstance(command.get("model_mode"), str) or not isinstance(command.get("decision_id"), str):
        raise ValueError("command needs model_mode and decision_id")
    finite(command.get("fps"), "command fps", positive=True)
    if type(command.get("valid_until_utc_ns")) is not int or command["valid_until_utc_ns"] <= 0:
        raise ValueError("command needs valid_until_utc_ns")
    return command


def validate_window_state(value: dict) -> dict:
    """Edge -> Cloud window summary for the scheduler: stream key, window id, current action, numeric raw state."""
    if not isinstance(value, dict) or value.get("control_v1") != VERSION:
        raise ValueError("invalid window state version")
    key = value.get("stream_key")
    if (not isinstance(key, list) or len(key) != 4 or not all(isinstance(x, str) and x for x in key[:3])
            or type(key[3]) is not int):
        raise ValueError("window state needs a stream_key [node, camera, stream, epoch]")
    if type(value.get("window_id")) is not int or value["window_id"] < 0:
        raise ValueError("invalid window_id")
    context = value.get("context")
    if not isinstance(context, dict) or not isinstance(context.get("model_mode"), str):
        raise ValueError("window state needs its current action")
    raw = value.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("window state needs raw measurements")
    for name, item in raw.items():
        if item is not None:
            finite(item, f"raw {name}")
    return value


def validate_result(reply, request):
    """Reject replay, mismatched stream/model/PTS and non-Hub completion."""
    if (reply.frame != request.frame or reply.model_mode != request.model_mode
            or reply.model_id != request.model_id or reply.request_id != request.request_id
            or reply.decision_id != request.decision_id or reply.preview_pts != request.preview_pts):
        raise ValueError("control reply frame/model/request/decision/PTS mismatch")
    attrs = reply.attributes
    expected = "probe_reply" if request.decision_id.startswith("probe/") else "result_hub_accepted"
    if attrs.get("control_v1") != VERSION or attrs.get("completion") != expected:
        raise ValueError(f"control reply is not a {expected} completion")
    finite(attrs.get("cloud_handling_ms"), "cloud_handling_ms")
    return reply
