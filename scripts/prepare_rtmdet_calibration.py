"""Representative float32 input list for SNPE; same preprocessing as Edge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--layout", choices=["nhwc", "nchw"], default="nhwc")
    args = parser.parse_args()
    if not 10 <= args.samples <= 1000 or args.output.exists():
        parser.error("samples must be 10..1000; output directory must be new")
    import cv2
    import numpy as np
    from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter
    from inference.model_artifact import sha256
    video = args.video.expanduser().resolve(strict=True)
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or total < args.samples:
            raise ValueError("video cannot supply calibration samples")
        args.output.mkdir(parents=True, exist_ok=False)
        adapter = object.__new__(RtmdetSmallSnpeAdapter)
        adapter.manifest = {"input_layout": args.layout}
        selected = np.linspace(0, total - 1, args.samples, dtype=np.int64)
        files = []
        for frame_id in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"cannot decode calibration frame {frame_id}")
            path = (args.output / f"frame_{frame_id:06d}.raw").resolve()
            if " " in str(path):
                raise ValueError("SNPE input list directory must not contain spaces")
            with path.open("xb") as target:
                target.write(adapter.prepare_frame(image).tensor.tobytes())
            files.append(str(path))
        with (args.output / "input_list.txt").open("x", encoding="utf-8") as target:
            target.write("\n".join(files) + "\n")
        with (args.output / "calibration.json").open("x", encoding="utf-8") as target:
            json.dump({"video_sha256": sha256(video), "frame_ids": selected.tolist(),
                       "input_layout": args.layout, "dtype": "float32", "samples": len(files),
                       "preprocessing": "MMDetection RTMDet640 keep-ratio BGR right-bottom pad114"}, target, indent=2)
        print(f"Prepared {len(files)} samples: {args.output / 'input_list.txt'}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
