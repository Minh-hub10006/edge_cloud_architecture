"""Normalized image-quality features used by the Edge window policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BlurDifficultyNormalizer:
    """Map Laplacian variance to 0=sharp/easy and 1=blurry/hard.

    Thresholds are camera/resolution dependent and must be selected from a
    calibration sample. This class deliberately has no guessed defaults.
    """

    blurry_variance: float
    sharp_variance: float

    def __post_init__(self) -> None:
        if self.blurry_variance < 0 or self.sharp_variance <= self.blurry_variance:
            raise ValueError("sharp_variance must be greater than non-negative blurry_variance")

    def difficulty(self, laplacian_variance: float) -> float:
        if laplacian_variance < 0:
            raise ValueError("laplacian_variance must be non-negative")
        fraction = (laplacian_variance - self.blurry_variance) / (
            self.sharp_variance - self.blurry_variance
        )
        return 1.0 - min(1.0, max(0.0, fraction))


def laplacian_variance(frame: Any) -> float:
    """Compute variance of the grayscale Laplacian for a decoded BGR frame."""

    import cv2
    import numpy as np

    pixels = np.asarray(frame)
    if pixels.ndim == 2:
        gray = pixels
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("frame must be a grayscale or BGR image")
    if gray.size == 0:
        raise ValueError("frame must not be empty")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def blur_difficulty(frame: Any, normalizer: BlurDifficultyNormalizer) -> float:
    return normalizer.difficulty(laplacian_variance(frame))
