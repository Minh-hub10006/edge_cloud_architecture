"""Scheduler policies (docs/PLAN_CONTROL_V3_VI.md C5): streams {key: window state} + Cloud telemetry ->
{key: (model_mode, fps)}. DRL v3 plugs in with the same signature."""
from __future__ import annotations


def stream_name(key):
    """node:camera:stream (the epoch changes every run, so assignments never name it)."""
    return ":".join(map(str, key[:3]))


class FixedPolicy:
    """Assignments from a file, e.g. {"default": ["yolo11s_edge", 15], "streams": {"edge-247": [...],
    "edge-246:traffic-01:control-cloud": [...]}}; a full stream name wins over its node. Replaces the old
    replay of the central DRL split with per-stream fixed models."""
    name = "fixed"

    def __init__(self, spec):
        self.default = tuple(spec["default"])
        self.streams = {k: tuple(v) for k, v in spec.get("streams", {}).items()}

    def decide(self, streams, cloud):
        out = {}
        for key in streams:
            mode, fps = self.streams.get(stream_name(key), self.streams.get(key[0], self.default))
            out[key] = (mode, float(fps))
        return out


class RulesPolicy:
    """Best hand rule of the live tables: the k streams on the strongest encoders run the accurate Cloud model,
    the rest the Edge model; k = floor(rho x GPU capacity / FPS), one less while the Cloud reports pressure."""
    name = "rules"

    def __init__(self, capacity_rps, *, cloud_mode="dfine_x_cloud", edge_mode="yolo11s_edge", fps=15.0, rho=.8):
        if capacity_rps <= 0:
            raise ValueError("rules policy needs the accurate model's measured capacity")
        self.capacity, self.cloud_mode, self.edge_mode = capacity_rps, cloud_mode, edge_mode
        self.fps, self.rho = fps, rho

    def decide(self, streams, cloud):
        k = int(self.rho * self.capacity // self.fps)
        pressure = None if cloud is None else cloud.get("cloud_arrival_pressure_norm")
        if pressure is not None and pressure >= .9:
            k = max(0, k - 1)
        def encoder(key):
            return float((streams[key].get("capability") or {}).get("encode_fps", 0.0))
        ranked = sorted(streams, key=lambda key: (-encoder(key), key))
        return {key: (self.cloud_mode if i < k else self.edge_mode, self.fps) for i, key in enumerate(ranked)}
