"""Browser-facing metadata adapters."""

from .metadata import serve_metadata_websocket
from .cloud_app import create_app, detection_result_from_payload
from .frame_sync import frame_pts_tolerance_ns, select_frame_metadata

__all__ = [
    "create_app", "detection_result_from_payload", "frame_pts_tolerance_ns",
    "select_frame_metadata", "serve_metadata_websocket",
]
