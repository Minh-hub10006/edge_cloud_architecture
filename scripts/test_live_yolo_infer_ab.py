"""Compare current and pre-overlap YOLO infer paths in an isolated live process.

Run from the deployed platform source root. The old-infer case restores only
YoloSnpeAdapter.infer. old-full also restores YOLO preprocessing. Publisher,
queues and Cloud remain the same. No installed source file is modified.
The owned worker has a watchdog.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time


def legacy_infer(self, bgr):
    """The infer method from playback_fix_20261002, without tensor copy/stages."""
    import numpy as np
    from inference.adapters import _normalize_xyxy

    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("YOLO input must be an HxWx3 BGR frame")
    started = time.perf_counter()
    raw = self.detector.detect_boxes(bgr)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    height, width = bgr.shape[:2]
    boxes = [box for item in raw if (box := _normalize_xyxy(item, width, height))]
    return boxes, elapsed_ms


def legacy_preprocess(self, frame_bgr):
    """The YOLO preprocessing method from playback_fix_20261002."""
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    r = min(self.imgsz / h, self.imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if self._pad is None:
        self._pad = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        if self.input_layout == "nhwc":
            self._tensor = np.empty((1, self.imgsz, self.imgsz, 3), dtype=np.float32)
        else:
            self._tensor = np.empty((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
    self._pad.fill(114)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    self._pad[:nh, :nw] = resized
    if self.input_layout == "nhwc":
        np.multiply(self._pad, 1.0 / 255.0, out=self._tensor[0], casting="unsafe")
    else:
        chw = self._pad.transpose(2, 0, 1)
        np.multiply(chw, 1.0 / 255.0, out=self._tensor[0], casting="unsafe")
    return self._tensor, r


@contextmanager
def infer_path(case):
    from inference.adapters import YoloSnpeAdapter
    from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector

    if case not in {"current", "old-infer", "old-full"}:
        raise ValueError("case must be current, old-infer or old-full")
    original = YoloSnpeAdapter.infer
    original_preprocess = YOLOSnpeDetector._preprocess
    if case in {"old-infer", "old-full"}:
        YoloSnpeAdapter.infer = legacy_infer
    if case == "old-full":
        YOLOSnpeDetector._preprocess = legacy_preprocess
    try:
        yield
    finally:
        YoloSnpeAdapter.infer = original
        YOLOSnpeDetector._preprocess = original_preprocess


def supervise(argv, source_root, timeout_s):
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--ab-worker", *argv]
    try:
        return subprocess.run(command, cwd=source_root, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        print("[YOLO A/B] TIMEOUT_INCOMPLETE: owned worker stopped", file=sys.stderr, flush=True)
        return 124
    except KeyboardInterrupt:
        return 130


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ab_parser = argparse.ArgumentParser(add_help=False)
    ab_parser.add_argument("--ab-path", choices=("current", "old-infer", "old-full"), default="current")
    ab_parser.add_argument("--source-root", type=Path, default=Path.cwd())
    ab_parser.add_argument("--ab-worker", action="store_true", help=argparse.SUPPRESS)
    ab, runner_argv = ab_parser.parse_known_args(argv)
    root = ab.source_root.expanduser().resolve()
    required = ("edge_agent/control_v1.py", "inference/adapters.py",
                "legacy/adapters/yolo/yolo_snpe.py")
    for name in required:
        if not (root / name).is_file():
            ab_parser.error(f"source root missing {name}: {root}")
    sys.path.insert(0, str(root))
    from edge_agent import control_v1
    args = control_v1.parser().parse_args(runner_argv)
    if args.mode != "yolo_edge" or args.policy != "observe":
        ab_parser.error("A/B requires --mode=yolo_edge --policy=observe")
    if not math.isfinite(args.run_timeout_s) or args.run_timeout_s <= 0:
        ab_parser.error("run timeout must be finite and positive")
    if not ab.ab_worker:
        print(f"[YOLO A/B] case={ab.ab_path} source={root}", flush=True)
        for name in required:
            digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
            print(f"[YOLO A/B] SHA256 {name} {digest}", flush=True)
        return supervise(argv, root, args.run_timeout_s)
    with infer_path(ab.ab_path):
        preprocessing = "playback" if ab.ab_path == "old-full" else "deployed"
        print(f"[YOLO A/B] worker case={ab.ab_path}; preprocess={preprocessing}; publisher unchanged", flush=True)
        asyncio.run(control_v1.run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
