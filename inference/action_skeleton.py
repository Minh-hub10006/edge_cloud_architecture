"""UC4 fight check on the Cloud (P3, 11/11): ST-GCN++ (MMAction2, Apache-2.0) on RTMO-s skeletons.

Input as MMAction2's test pipeline for stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d (PreNormalize2D,
UniformSampleFrames clip_len 100 one clip, FormatGCNInput 2 persons zero-padded): (1, 2, 100, 17, 3) - x, y in [-1, 1]
of the image the skeletons were found in, keypoint score. A short clip (Edge triggers send ~2 s = 10-20 frames) is looped
to 100 frames, as that pipeline does. Output: softmax over NTU60.
Weights: NTU60 for now (11/11: on 30 surveillance fight clips AUC 0.65-0.70 - to retrain on fight clips later).
"""
from __future__ import annotations

import numpy as np

CLIP_LEN, PERSONS = 100, 2
FIGHT_CLASSES = (49, 50, 51)   # NTU60: punching/slapping, kicking, pushing other person


def prepare_clip(frames, image_hw) -> np.ndarray:
    """frames: per frame an array (people, 17, 3) [x, y, score] in pixels, people in a stable order (by track) ->
    float32 (1, 2, 100, 17, 3)."""
    h, w = image_hw
    n = len(frames)
    x = np.zeros((PERSONS, max(n, 1), 17, 3), np.float32)
    seen = 0
    for t, people in enumerate(frames):
        people = np.asarray(people, np.float32).reshape(-1, 17, 3)[:PERSONS]
        x[:len(people), t] = people
        seen = max(seen, len(people))
    # as MMAction2: people slots used anywhere in the clip are normalized (an empty frame there becomes -1, -1, score 0),
    # unused slots stay all 0
    x[:seen, ..., 0] = (x[:seen, ..., 0] - w / 2) / (w / 2)
    x[:seen, ..., 1] = (x[:seen, ..., 1] - h / 2) / (h / 2)
    # shortcut: loops clips shorter than 100 frames like MMAction2; longer ones (> 10 s at 10 fps) are cut, not sampled
    return x[:, np.arange(CLIP_LEN) % max(n, 1)][None]


def fight_score(prob) -> float:
    return float(np.asarray(prob).reshape(-1)[list(FIGHT_CLASSES)].sum())
