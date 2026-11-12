"""DRL state v2 (22 values) and the trained-actor runtime: numpy MLP only, no torch on the Edge.

Single source of the observation layout for simulator, trainer and live control (docs/DRL_DESIGN_VI.md).
Each value is the mean over the last HISTORY one-second windows; missing optional measurements are 0
with an availability bit. Actor file (JSON, format "edge-drl-actor-v2"):
  {"format": ..., "observation": [OBS_V2...], "actions": [mode, ...],
   "layers": [{"w": [[in x out]], "b": [out], "act": "tanh"|"relu"|"linear"}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "edge-drl-actor-v2"
HISTORY = 5  # windows aggregated per decision (decision every 5 s on 1 s windows)
# shortcut: v6 actors and drl/sim.py still name LW-DETR; DRL v3 retrains with dfine_x_cloud.
ACTION_MODES = ("yolo_edge", "yolo11s_edge", "rtdetr_cloud", "lwdetr_large_cloud")
OBS_V2 = ("cap_yolov8n", "cap_yolo11s", "cap_encode",
          "cpu_max_core", "temp", "temp_ok", "edge_backlog", "encoder_load",
          "transport", "transport_ok", "cloud_pressure",
          "timely_ratio", "path_latency",
          "brightness", "motion", "vehicle_count", "box_change", "box_change_ok",
          *(f"mode_{m}" for m in ACTION_MODES))
CAP_SCALE_FPS = 300.0
_ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0), "linear": lambda x: x}


def _mean(snapshots, name):
    values = [s["raw"].get(name) for s in snapshots]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _clip(value):
    return min(1.0, max(0.0, float(value)))


def observation_v2(snapshots, current_mode) -> np.ndarray:
    """snapshots: recent TimeWindowMonitor.finish() outputs (oldest first); current_mode: mode value."""
    if not snapshots:
        raise ValueError("observation needs at least one window")
    if current_mode not in ACTION_MODES:
        raise ValueError(f"unknown mode {current_mode}")
    recent = list(snapshots)[-HISTORY:]
    last = recent[-1]
    capability = last.get("capability") or {}
    missing = [k for k in ("yolov8n_fps", "yolo11s_fps", "encode_fps") if k not in capability]
    if missing:
        raise ValueError(f"device capability missing {missing}")
    target = last["context"]["target_fps"]
    media_fps = last.get("media_fps") or target

    def optional(name, scale):
        value = _mean(recent, name)
        return (0.0, 0.0) if value is None else (_clip(value / scale), 1.0)

    temp, temp_ok = optional("temp_c", 100.0)
    transport, transport_ok = optional("transport_overhead_ms", 100.0)
    box_change, box_change_ok = optional("box_change_mean", 1.0)
    encoder = _mean(recent, "encoder_write_ms")
    timely = _mean(recent, "timely_fps")
    latency = _mean(recent, "active_path_latency_ms")
    values = [
        _clip(capability["yolov8n_fps"] / CAP_SCALE_FPS), _clip(capability["yolo11s_fps"] / CAP_SCALE_FPS),
        _clip(capability["encode_fps"] / CAP_SCALE_FPS),
        _clip((_mean(recent, "cpu_max_core_pct") or 0.0) / 100), temp, temp_ok,
        _clip(_mean(recent, "edge_backlog_norm") or 0.0),
        0.0 if encoder is None else _clip(encoder * media_fps / 1000),
        transport, transport_ok, _clip(_mean(recent, "cloud_arrival_pressure_norm") or 0.0),
        _clip((timely or 0.0) / target),
        # Nothing completed in the whole history is the worst latency, not an unknown-good one.
        1.0 if latency is None else _clip(latency / 1000),
        _clip((_mean(recent, "brightness") or 0.0) / 255), _clip((_mean(recent, "motion") or 0.0) * 4 / 255),
        _clip((_mean(recent, "vehicle_count") or 0.0) / 50), box_change, box_change_ok,
        *(1.0 if m == current_mode else 0.0 for m in ACTION_MODES)]
    return np.array(values, dtype=np.float64)


class MlpActor:
    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT:
            raise ValueError(f"actor format must be {FORMAT}")
        if tuple(spec.get("observation", ())) != OBS_V2:
            raise ValueError("actor observation layout differs from OBS_V2")
        actions = tuple(spec.get("actions") or ())
        if not actions or len(set(actions)) != len(actions) or any(a not in ACTION_MODES for a in actions):
            raise ValueError("actor actions must be distinct known modes")
        self.actions = actions
        self.layers = []
        width = len(OBS_V2)
        for layer in spec.get("layers") or ():
            w, b = np.asarray(layer["w"], dtype=np.float64), np.asarray(layer["b"], dtype=np.float64)
            act = layer.get("act", "linear")
            if act not in _ACTIVATIONS or w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                raise ValueError(f"invalid actor layer shape/activation at input width {width}")
            if not (np.isfinite(w).all() and np.isfinite(b).all()):
                raise ValueError("actor weights must be finite")
            self.layers.append((w, b, _ACTIVATIONS[act]))
            width = w.shape[1]
        if not self.layers or width != len(self.actions):
            raise ValueError("actor output width must equal the number of actions")

    @classmethod
    def load(cls, path) -> "MlpActor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def logits(self, observation) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        for w, b, act in self.layers:
            x = act(x @ w + b)
        return x

    def act(self, observation, mask) -> int:
        """Deterministic greedy action among unmasked ones (evaluation/live, not exploration)."""
        if len(mask) != len(self.actions) or not any(mask):
            raise ValueError("mask must cover every action and allow at least one")
        logits = self.logits(observation)
        if not np.isfinite(logits).all():
            raise ValueError("actor produced non-finite logits")
        return int(max((i for i, ok in enumerate(mask) if ok), key=lambda i: (logits[i], -i)))


def write_actor(path, actions, layers) -> None:
    """Trainer helper: actions [mode], layers [(w, b, act)] -> validated JSON file."""
    spec = {"format": FORMAT, "observation": list(OBS_V2), "actions": list(actions),
            "layers": [{"w": np.asarray(w).tolist(), "b": np.asarray(b).tolist(), "act": a} for w, b, a in layers]}
    MlpActor(spec)  # refuse to write an actor the Edge would reject
    Path(path).write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
