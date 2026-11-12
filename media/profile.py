"""Validated low-latency preview profile and per-stream URLs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit

StreamKey = tuple[str, str, str, int]
_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class MediaProfile:
    width: int = 640
    height: int = 360
    fps: int = 15
    bitrate_kbps: int = 800
    gop_frames: int = 15

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("YUV420 preview dimensions must be positive and even")
        if self.fps <= 0 or self.bitrate_kbps <= 0 or self.gop_frames <= 0:
            raise ValueError("fps, bitrate and GOP must be positive")


def media_path(stream_key: StreamKey) -> str:
    node_id, camera_id, stream_id, epoch = stream_key
    if not all(_SEGMENT.fullmatch(value) for value in (node_id, camera_id, stream_id)):
        raise ValueError("node, camera and stream IDs must use letters, digits, _ or -")
    if epoch < 0:
        raise ValueError("stream epoch must be non-negative")
    return f"edge/{node_id}/{camera_id}/{stream_id}/epoch-{epoch}"


def _base_url(base: str, scheme: str) -> str:
    parts = urlsplit(base)
    if parts.scheme != scheme or not parts.hostname or parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError(f"expected a {scheme} base URL containing only host and port")
    return base.rstrip("/")


def rtsp_publish_url(rtsp_base: str, stream_key: StreamKey) -> str:
    return f"{_base_url(rtsp_base, 'rtsp')}/{media_path(stream_key)}"


def webrtc_page_url(http_base: str, stream_key: StreamKey) -> str:
    return f"{_base_url(http_base, 'http')}/{media_path(stream_key)}"
