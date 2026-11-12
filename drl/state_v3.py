"""Per-stream state vector shared by the simulator and the live Cloud scheduler (docs/PLAN_CONTROL_V3_VI.md C6).

Input: the stream's latest window `raw` dict (live: Edge window state; sim: MultiEdgeSim snapshot) and its current
action. Three variants (docs/PLAN_PROBE_VI.md): "probe" (A), "image" (B), "none" (C, resources only). Missing values
become 0 with an availability bit per group, so a trained actor sees the same vector in both worlds.
"""
from __future__ import annotations

import math

MODES = ("yolo_edge", "yolo11s_edge", "rtdetr_cloud", "dfine_x_cloud")
# (raw name, scale): value / scale clipped to [0, 1]
RESOURCES = (("cpu_max_core_pct", 100), ("temp_c", 100), ("encoder_write_ms", 1000 / 15),
             ("edge_backlog_norm", 1), ("active_path_latency_ms", 500), ("timely_fps", 15),
             ("transport_overhead_ms", 100), ("cloud_arrival_pressure_norm", 1))
PROBE = (("probe_missed_last", 10), ("probe_missed", 10), ("probe_age_s", 30))
IMAGE = (("brightness", 255), ("contrast", 128), ("laplacian_small", None), ("motion", 64),
         ("vehicle_count", 30), ("small_ratio", 1), ("conf_mean", 1), ("uncertain_ratio", 1))
VARIANTS = ("probe", "image", "none")


def _scaled(raw, name, scale):
    value = raw.get(name)
    if value is None:
        return None
    if scale is None:  # Laplacian variance spans orders of magnitude
        return min(1.0, math.log1p(max(0.0, value)) / 10)
    return min(1.0, max(0.0, value / scale))


def _group(raw, spec):
    values = [_scaled(raw, name, scale) for name, scale in spec]
    return [0.0 if v is None else v for v in values] + [float(any(v is not None for v in values))]


def names(variant):
    out = [n for n, _ in RESOURCES] + ["resources_available"]
    if variant == "probe":
        out += [n for n, _ in PROBE] + ["probe_available"]
    elif variant == "image":
        out += [n for n, _ in IMAGE] + ["image_available"]
    elif variant != "none":
        raise ValueError(f"state variant must be one of {VARIANTS}")
    return out + [f"mode_{m}" for m in MODES] + ["fps"]


def vector(raw, mode, fps, variant):
    """raw: one window's measurements; mode/fps: the stream's current action."""
    out = _group(raw, RESOURCES)
    if variant == "probe":
        out += _group(raw, PROBE)
        if raw.get("probe_age_s") is None:  # never probed: the oldest possible information
            out[-2] = 1.0
    elif variant == "image":
        out += _group(raw, IMAGE)
    elif variant != "none":
        raise ValueError(f"state variant must be one of {VARIANTS}")
    return out + [float(mode == m) for m in MODES] + [min(1.0, fps / 15)]
