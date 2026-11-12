"""Per-camera slow-lane profile (P1, 10/11): which use cases run, their thresholds, and the zones they look at.

  {"default": {...}, "edge-246/cam-1": {"tasks": ["plates", "congestion", "count"],
                                        "config": {"congestion_vehicles": 25},
                                        "zones": {"road": [[0.1, 0.4], [0.9, 0.4], [1, 1], [0, 1]]}}}

A camera's entry is merged over "default" (tasks replace, config and zones merge). Zones are polygons in 0..1 frame
coordinates; an object is inside when the middle of its box bottom (where it touches the ground) is. Without a zone the
whole frame counts. Zone names used by the tasks: "road" (UC2 congestion), "area" (UC3 counting), "watch" (UC3
gathering, UC4 loitering).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_profile(path: str | Path, node_id: str, camera_id: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("camera profile file must be a JSON object")
    base, own = data.get("default", {}), data.get(f"{node_id}/{camera_id}", {})
    profile = {"tasks": own.get("tasks", base.get("tasks", [])),
               "config": {**base.get("config", {}), **own.get("config", {})},
               "zones": {**base.get("zones", {}), **own.get("zones", {})}}
    for name, polygon in profile["zones"].items():
        if (not isinstance(polygon, list) or len(polygon) < 3
                or not all(len(p) == 2 and all(0 <= v <= 1 for v in p) for p in polygon)):
            raise ValueError(f"zone {name!r} of {node_id}/{camera_id}: need >= 3 [x, y] points in 0..1")
    return profile


def inside(point, polygon) -> bool:
    """Even-odd ray casting; points on an edge may fall either way (fine for 0..1 ground points)."""
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def in_zone(box, polygon) -> bool:
    """None = no zone configured = everywhere."""
    return polygon is None or inside(((box.left + box.right) / 2, box.bottom), polygon)
