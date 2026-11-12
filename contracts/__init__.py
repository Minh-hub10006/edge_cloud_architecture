"""Stable messages and interfaces shared by Edge and Cloud services."""

from .messages import (
    Box,
    CloudInferenceTicket,
    DetectionResult,
    FrameIdentity,
    MediaFrameRef,
    ModelMode,
    Placement,
    RouteState,
    WindowStateFeedback,
)
from .confidence_calibration import PlattConfidenceCalibrator, calibrate_detection_result
from .model_registry import DEFAULT_MODEL_REGISTRY, ModelProfile
from .in_memory_transport import InMemoryTransport
from .transport import TransportAdapter, TransportEnvelope, TransportKind

__all__ = [
    "Box",
    "PlattConfidenceCalibrator",
    "calibrate_detection_result",
    "CloudInferenceTicket",
    "DEFAULT_MODEL_REGISTRY",
    "DetectionResult",
    "FrameIdentity",
    "MediaFrameRef",
    "InMemoryTransport",
    "ModelMode",
    "ModelProfile",
    "Placement",
    "RouteState",
    "TransportAdapter",
    "TransportEnvelope",
    "TransportKind",
    "WindowStateFeedback",
]
