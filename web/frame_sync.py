"""Frame-PTS matching helpers shared by overlay tests and browser policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def frame_pts_tolerance_ns(fps: float) -> int:
    """Maximum absolute PTS error for a presentation-frame match (half frame)."""

    if not 0 < fps <= 240:
        raise ValueError("fps must be in (0, 240]")
    return round(500_000_000 / fps)


def select_frame_metadata(
    presented_pts_ns: int,
    metadata_by_pts: Mapping[int, Any],
    *,
    fps: float,
) -> tuple[int, Any] | None:
    """Return the closest metadata event only when its PTS identifies this frame."""

    if presented_pts_ns < 0:
        raise ValueError("presented_pts_ns must be non-negative")
    tolerance = frame_pts_tolerance_ns(fps)
    if not metadata_by_pts:
        return None
    matched_pts = min(metadata_by_pts, key=lambda pts: abs(pts - presented_pts_ns))
    if abs(matched_pts - presented_pts_ns) > tolerance:
        return None
    return matched_pts, metadata_by_pts[matched_pts]
