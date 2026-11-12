"""Frame preprocess for D-FINE Segment A (640×640)."""

from __future__ import annotations

import cv2
import numpy as np

from detectors.dfine.tier1_codec import INPUT_SHAPE

H, W = INPUT_SHAPE[2], INPUT_SHAPE[3]


def frame_to_images_tensor(frame: np.ndarray) -> np.ndarray:
    """
    BGR uint8 frame → float32 NCHW [1, 3, 640, 640].
    Match pipeline: resize 640×640; normalize to [0,1] (extend with mean/std if training config requires).
    """
    img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)
