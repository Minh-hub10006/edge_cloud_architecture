"""Explicit per-model confidence calibration; raw scores are never relabeled."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .messages import DetectionResult


@dataclass(frozen=True, slots=True)
class PlattConfidenceCalibrator:
    """Platt mapping fitted offline against labeled detection correctness.

    ``slope`` and ``intercept`` must come from a held-out calibration set for
    this exact model/artifact and evaluation domain. The runtime intentionally
    does not guess these parameters.
    """

    model_id: str
    profile_id: str
    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if not self.model_id or not self.profile_id:
            raise ValueError("model_id and profile_id are required")
        if not math.isfinite(self.slope) or not math.isfinite(self.intercept):
            raise ValueError("calibration parameters must be finite")

    def apply(self, raw_confidence: float) -> float:
        if not 0.0 <= raw_confidence <= 1.0:
            raise ValueError("raw confidence must be in [0, 1]")
        eps = 1e-7
        probability = min(1.0 - eps, max(eps, raw_confidence))
        logit = math.log(probability / (1.0 - probability))
        value = self.slope * logit + self.intercept
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)


def calibrate_detection_result(
    result: DetectionResult, calibrator: PlattConfidenceCalibrator,
) -> DetectionResult:
    """Return a calibrated copy, requiring the profile to match the model."""

    if calibrator.model_id != result.model_id:
        raise ValueError("calibration profile belongs to another model_id")
    boxes = tuple(
        replace(box, calibrated_confidence=calibrator.apply(box.confidence))
        for box in result.boxes
    )
    return replace(
        result,
        boxes=boxes,
        confidence_calibration_id=calibrator.profile_id,
    )
