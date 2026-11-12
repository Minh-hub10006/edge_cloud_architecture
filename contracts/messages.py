"""Versioned domain messages for the new platform.

Coordinates in ``Box`` are normalized to the source frame in [0, 1].
Cloud does not decide a route: only Edge creates ``RouteState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


# Pre-deployment contract update: state feedback now uses blur difficulty and
# calibrated confidence instead of box count and raw confidence.
SCHEMA_VERSION = 2
MAX_TICKET_JPEG_BYTES = 1 << 20  # a 640 px q95 frame is ~100 KB
MAX_TICKET_TENSOR_BYTES = 2_000_000  # path B cut tensor (contracts/cut_tensor.py), ~450 KB; Kafka topics allow 2 MiB


class Placement(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


class ModelMode(str, Enum):
    YOLO_EDGE = "yolo_edge"
    RTMDET_EDGE = "rtmdet_edge"
    DFINE_CLOUD = "dfine_cloud"
    RTDETR_CLOUD = "rtdetr_cloud"
    LWDETR_CLOUD = "lwdetr_large_cloud"
    YOLO11S_EDGE = "yolo11s_edge"
    DFINEX_CLOUD = "dfine_x_cloud"
    # Path B (v4): Edge NPU runs RT-DETRv2-S up to layer2, the ticket carries the cut tensor, Cloud runs the tail.
    RTDETR_SPLIT_CLOUD = "rtdetr_split_cloud"

    @property
    def placement(self) -> Placement:
        return Placement.EDGE if self in {
            ModelMode.YOLO_EDGE,
            ModelMode.RTMDET_EDGE,
            ModelMode.YOLO11S_EDGE,
        } else Placement.CLOUD


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Identity of a source frame across Edge, media and Cloud services."""

    node_id: str
    camera_id: str
    stream_id: str
    stream_epoch: int
    source_frame_id: int
    capture_utc_ns: int
    source_pts: int

    def __post_init__(self) -> None:
        if not all((self.node_id, self.camera_id, self.stream_id)):
            raise ValueError("node_id, camera_id and stream_id are required")
        if self.stream_epoch < 0:
            raise ValueError("stream_epoch must be non-negative")
        if self.source_frame_id < 0:
            raise ValueError("source_frame_id must be non-negative")
        if self.capture_utc_ns <= 0:
            raise ValueError("capture_utc_ns must be positive")
        if self.source_pts < 0:
            raise ValueError("source_pts must be non-negative")

    @property
    def stream_key(self) -> tuple[str, str, str, int]:
        return (self.node_id, self.camera_id, self.stream_id, self.stream_epoch)


@dataclass(frozen=True, slots=True)
class MediaFrameRef:
    """Small sidecar binding one encoded H.264 frame to its source identity.

    encoded_seq and media_pts_ns identify the frame after the preview cadence
    gate. A reference is emitted only after the Edge publisher accepts the raw
    frame, so skipped or queue-dropped source frames get no reference.
    """

    frame: FrameIdentity
    encoded_seq: int
    media_pts_ns: int
    submitted_utc_ns: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported media frame reference schema_version")
        if self.encoded_seq < 0 or self.media_pts_ns < 0:
            raise ValueError("encoded_seq and media_pts_ns must be non-negative")
        if self.submitted_utc_ns <= 0:
            raise ValueError("submitted_utc_ns must be positive")


@dataclass(frozen=True, slots=True)
class Box:
    """A detector box in source-frame normalized coordinates."""

    left: float
    top: float
    right: float
    bottom: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None
    calibrated_confidence: float | None = None

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom, self.confidence)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized coordinates and confidence must be in [0, 1]")
        if self.left > self.right or self.top > self.bottom:
            raise ValueError("box coordinates are inverted")
        if not self.class_name:
            raise ValueError("class_name is required")
        if self.calibrated_confidence is not None and not 0.0 <= self.calibrated_confidence <= 1.0:
            raise ValueError("calibrated_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Normalized result consumed by Cloud Result Hub and ByteTrack."""

    frame: FrameIdentity
    model_mode: ModelMode
    model_id: str
    decision_id: str
    boxes: Sequence[Box]
    inference_ms: float
    produced_utc_ns: int
    request_id: str | None = None
    preview_pts: int | None = None
    schema_version: int = SCHEMA_VERSION
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    confidence_calibration_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported detection schema_version")
        if not self.model_id or not self.decision_id:
            raise ValueError("model_id and decision_id are required")
        if self.inference_ms < 0:
            raise ValueError("inference_ms must be non-negative")
        if self.produced_utc_ns <= 0:
            raise ValueError("produced_utc_ns must be positive")
        if self.preview_pts is not None and self.preview_pts < 0:
            raise ValueError("preview_pts must be non-negative")
        calibrated = [box.calibrated_confidence is not None for box in self.boxes]
        if any(calibrated) and not all(calibrated):
            raise ValueError("calibrated confidence must be present for every box or none")
        if any(calibrated) and not self.confidence_calibration_id:
            raise ValueError("confidence_calibration_id is required for calibrated boxes")


@dataclass(frozen=True, slots=True)
class CloudInferenceTicket:
    """Small Edge-authored request to infer one frame: carried as ``jpeg``, or (empty jpeg)
    already present in the Cloud RTSP media cache.

    preview_pts is the H.264 media PTS in nanoseconds, not the source PTS.
    Without jpeg the Cloud cache verifies it against the matching MediaFrameRef.
    """

    frame: FrameIdentity
    model_mode: ModelMode
    model_id: str
    decision_id: str
    preview_pts: int
    deadline_utc_ns: int
    request_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    # Frame pixels for Cloud (CLOUD_FRAME_SOURCE=jpeg); empty = Cloud joins its RTSP frame cache.
    jpeg: bytes = field(default=b"", repr=False)
    # Path B only: the packed cut tensor (contracts/cut_tensor.py) instead of pixels.
    tensor: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported ticket schema_version")
        if self.jpeg and (len(self.jpeg) > MAX_TICKET_JPEG_BYTES or not self.jpeg.startswith(b"\xff\xd8")):
            raise ValueError("ticket jpeg must be a JPEG of at most 1 MiB")
        if self.tensor and self.jpeg:
            raise ValueError("ticket carries either jpeg or tensor, not both")
        if (self.model_mode is ModelMode.RTDETR_SPLIT_CLOUD) != bool(self.tensor):
            raise ValueError("rtdetr_split_cloud tickets (and only they) carry a cut tensor")
        if len(self.tensor) > MAX_TICKET_TENSOR_BYTES:
            raise ValueError("ticket tensor exceeds 2 MB")
        if self.model_mode.placement is not Placement.CLOUD:
            raise ValueError("CloudInferenceTicket requires a cloud model mode")
        if not self.model_id or not self.decision_id:
            raise ValueError("model_id and decision_id are required")
        if self.preview_pts < 0 or self.deadline_utc_ns <= 0:
            raise ValueError("preview_pts and deadline_utc_ns are invalid")


@dataclass(frozen=True, slots=True)
class WindowStateFeedback:
    """Cloud measurement summary for one window; never a route command.

    Blur difficulty is 0=sharp/easy and 1=blurry/hard. Calibrated confidence
    remains None when the window contains no usable calibrated detections.
    """

    stream_key: tuple[str, str, str, int]
    window_id: int
    decision_id: str
    model_id: str
    first_frame_id: int
    last_frame_id: int
    feedback_utc_ns: int
    blur_difficulty_mean: float
    calibrated_confidence_mean: float | None
    box_change_mean: float
    tracking_stability: float
    cloud_queue_depth: int
    cloud_p95_inference_ms: float
    cloud_gpu_utilization_pct: float
    transport_rtt_ms: float
    drop_ratio: float
    media_bitrate_mbps: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported feedback schema_version")
        if not self.decision_id or not self.model_id:
            raise ValueError("decision_id and model_id are required")
        if self.window_id < 0 or self.first_frame_id < 0 or self.last_frame_id < self.first_frame_id:
            raise ValueError("invalid window frame range")
        if self.feedback_utc_ns <= 0 or self.cloud_queue_depth < 0:
            raise ValueError("invalid feedback metadata")
        bounded = (self.blur_difficulty_mean, self.box_change_mean, self.tracking_stability, self.drop_ratio)
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("normalized feedback values must be in [0, 1]")
        if self.calibrated_confidence_mean is not None and not 0.0 <= self.calibrated_confidence_mean <= 1.0:
            raise ValueError("calibrated_confidence_mean must be in [0, 1] or None")
        if any(value < 0 for value in (
            self.cloud_p95_inference_ms,
            self.cloud_gpu_utilization_pct,
            self.transport_rtt_ms,
            self.media_bitrate_mbps,
        )):
            raise ValueError("feedback metrics must be non-negative")

    @property
    def confidence_difficulty_mean(self) -> float | None:
        """Return the policy-oriented confidence feature (0=confident, 1=uncertain)."""

        if self.calibrated_confidence_mean is None:
            return None
        return 1.0 - self.calibrated_confidence_mean


@dataclass(frozen=True, slots=True)
class RouteState:
    """Edge-authored decision committed for the next window."""

    stream_key: tuple[str, str, str, int]
    window_id: int
    decision_id: str
    model_mode: ModelMode
    target_fps: float
    effective_from_frame_id: int
    committed_utc_ns: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported route schema_version")
        if self.window_id < 0 or self.effective_from_frame_id < 0:
            raise ValueError("window and frame IDs must be non-negative")
        if not self.decision_id or self.target_fps <= 0:
            raise ValueError("decision_id is required and target_fps must be positive")
        if self.committed_utc_ns <= 0:
            raise ValueError("committed_utc_ns must be positive")
