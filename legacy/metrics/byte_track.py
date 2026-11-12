from __future__ import annotations

import math
import numpy as np
import torch

from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

# person, car, motorcycle, bus, truck — VN traffic + plate association
DEFAULT_TRACK_CLASSES = [0, 2, 3, 5, 7]


class _DetectionContainer:
    def __init__(self, boxes: np.ndarray):
        self._boxes = boxes
        n = len(boxes)
        self._xywh = np.zeros((n, 4), dtype=np.float32)
        if n:
            self._xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
            self._xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
            self._xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            self._xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        # Cache once: BYTETracker reads these properties many times per update.
        self._conf_t = torch.from_numpy(self._boxes[:, 4]) if n else torch.zeros(0, dtype=torch.float32)
        self._xywh_t = torch.from_numpy(self._xywh)
        self._cls_t = (
            torch.from_numpy(self._boxes[:, 5].astype(np.int64))
            if n
            else torch.zeros(0, dtype=torch.int64)
        )

    @property
    def conf(self):
        return self._conf_t

    @property
    def xywh(self):
        return self._xywh_t

    @property
    def cls(self):
        return self._cls_t

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        if isinstance(idx, np.ndarray) and idx.dtype == bool:
            new_boxes = self._boxes[idx]
        else:
            new_boxes = self._boxes[idx]
        return _DetectionContainer(new_boxes)


class EdgeTracker:
    def __init__(self, frame_rate: int = 30):
        self.args = IterableSimpleNamespace(
            track_high_thresh=0.3,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=60,
            match_thresh=0.5,
            fuse_score=True,
        )
        self.frame_rate = frame_rate
        self.tracker = BYTETracker(self.args)

    def reset(self):
        self.tracker = BYTETracker(self.args)

    def update(
        self,
        boxes_list: list[dict],
        img: np.ndarray,
        keep_classes: list[int] | None = None,
    ) -> list[dict]:
        height, width = img.shape[:2]
        valid_boxes = []
        for box in boxes_list:
            try:
                left = max(0.0, min(float(width), float(box["left"])))
                top = max(0.0, min(float(height), float(box["top"])))
                right = max(0.0, min(float(width), float(box["right"])))
                bottom = max(0.0, min(float(height), float(box["bottom"])))
                confidence = float(box.get("confidence", 0.5))
                class_id = int(box.get("class_id", 0))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(v) for v in (left, top, right, bottom, confidence)):
                continue
            if right - left <= 1e-6 or bottom - top <= 1e-6:
                continue
            if keep_classes is not None and class_id not in keep_classes:
                continue
            clean = dict(box)
            clean.update(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                confidence=confidence,
                class_id=class_id,
            )
            valid_boxes.append(clean)
        boxes_list = valid_boxes

        if not boxes_list:
            self.tracker.update(_DetectionContainer(np.zeros((0, 6), dtype=np.float32)), img)
            return []

        dets = []
        for b in boxes_list:
            x1 = float(b["left"])
            y1 = float(b["top"])
            x2 = float(b["right"])
            y2 = float(b["bottom"])
            score = float(b.get("confidence", 0.5))
            cls_id = int(b.get("class_id", 0))
            dets.append([x1, y1, x2, y2, score, cls_id])

        dets_np = np.array(dets, dtype=np.float32)
        container = _DetectionContainer(dets_np)
        tracked = self.tracker.update(container, img)

        if tracked is None or len(tracked) == 0:
            return boxes_list

        track_map = {}
        for row in tracked:
            _, _, _, _, track_id, score, cls_id, orig_idx = row
            track_map[int(orig_idx)] = int(track_id)

        out = []
        for i, b in enumerate(boxes_list):
            track_id = track_map.get(i, -1)
            b = dict(b)
            b["track_id"] = track_id
            out.append(b)
        return out
