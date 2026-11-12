"""Cheap Edge resource and scene measurements for the control state (no labels, no GPU)."""
from __future__ import annotations

import glob

import numpy as np

SCENE_SIZE = (160, 90)  # width, height of the downscaled frame used for pixel features
VEHICLE_CLASSES = frozenset({1, 2, 3, 5, 7})  # COCO bicycle, car, motorcycle, bus, truck
SMALL_AREA = (32 / 1280) * (32 / 720)  # COCO "small" (32x32 px) at the 1280x720 media profile


class ResourceSampler:
    """psutil CPU/RAM plus the hottest readable thermal zone; temp is None where unreadable (Windows)."""

    def __init__(self, thermal_glob="/sys/class/thermal/thermal_zone*/temp"):
        import psutil
        self.psutil = psutil
        self.thermal_glob = thermal_glob
        psutil.cpu_percent(percpu=True)  # prime: the first call has no interval to measure

    def temperature_c(self):
        values = []
        for path in glob.glob(self.thermal_glob):
            try:
                with open(path, encoding="ascii") as file:
                    values.append(int(file.read().strip()) / 1000)
            except (OSError, ValueError):
                continue  # disabled sensors return EINVAL
        return max(values) if values else None

    def sample(self) -> dict:
        per_core = self.psutil.cpu_percent(percpu=True)
        return {"cpu_max_core_pct": max(per_core) if per_core else None,
                "cpu_total_pct": sum(per_core) / len(per_core) if per_core else None,
                "ram_pct": self.psutil.virtual_memory().percent, "temp_c": self.temperature_c()}


def scene_features(bgr, previous_gray=None):
    """-> (features, gray): brightness/contrast/laplacian of a 160x90 gray copy, motion vs previous."""
    import cv2
    gray = cv2.cvtColor(cv2.resize(bgr, SCENE_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    features = {"brightness": float(gray.mean()), "contrast": float(gray.std()),
                "laplacian_small": float(cv2.Laplacian(gray, cv2.CV_64F).var())}
    if previous_gray is not None and previous_gray.shape == gray.shape:
        features["motion"] = float(np.abs(gray.astype(np.int16) - previous_gray.astype(np.int16)).mean())
    return features, gray


def box_features(boxes):
    """Normalized boxes (objects with class_id/left/top/right/bottom/confidence) -> vehicle count, small ratio and
    the confidence features of state variant B, defined as in scripts/hardness_signal.py (boxes >= 0.25)."""
    vehicles = [b for b in boxes if int(b.class_id) in VEHICLE_CLASSES]
    small = sum((b.right - b.left) * (b.bottom - b.top) < SMALL_AREA for b in vehicles)
    sure = [c for c in (getattr(b, "confidence", None) for b in vehicles) if c is not None and c >= .25]
    return {"vehicle_count": len(vehicles), "small_ratio": small / len(vehicles) if vehicles else None,
            "conf_mean": sum(sure) / len(sure) if sure else None,
            "uncertain_ratio": sum(c < .5 for c in sure) / len(sure) if sure else None}
