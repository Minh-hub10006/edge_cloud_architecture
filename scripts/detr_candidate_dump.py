"""Offline per-frame detections of a D-FINE / DEIM / DEIMv2 ONNX export over a folder of videos, in the
edge_parity dump format scored by detrac.py (table / ap):

  python -B scripts/detr_candidate_dump.py MODEL.onnx VIDEO_DIR OUT_DIR [--crop-bottom 64] [--batch 8]
      [--confidence 0.05] [--labels auto|coco91|contiguous] [--normalize none|imagenet] [--limit-frames N]
      [--jpeg-quality Q [--jpeg-size 640]]

--jpeg-quality feeds what a v50 Cloud model receives: the Edge's square INTER_AREA resize, then a JPEG round trip
with the Edge encoder (edge_agent.control_v1.encode_cloud_jpeg); 0 = the square resize without JPEG (baseline).

Also LW-DETR official exports (input "input" [1,3,640,640] ImageNet-normalized; dets [1,300,4] normalized cxcywh,
labels [1,300,91] logits; query x class top-k 300 as inference/detection_decode.lwdetr_decode), batch 1.

Contract of the official D-FINE family exports (tools/deployment/export_onnx.py): inputs images [N,3,640,640] float RGB /255
(DEIMv2 DINOv3 variants: ImageNet mean/std, --normalize imagenet) and orig_target_sizes [N,2] int64 (w, h);
outputs labels / boxes (xyxy pixels of the original size) / scores. COCO configs remap labels to COCO category
ids; the dump stores contiguous 0..79 ids as the existing RT-DETRv2 / LW-DETR dumps do. Resumable per video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

# COCO category id (1..90 with gaps) -> contiguous 0..79
COCO91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
          33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
          60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
TO_CONTIGUOUS = {c: i for i, c in enumerate(COCO91)}
MEAN = np.array([.485, .456, .406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([.229, .224, .225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(frames, normalize, size=640):
    import cv2
    x = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_LINEAR)
                  for f in frames]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return (x - MEAN) / STD if normalize == "imagenet" else x


def to_contiguous(label, mode):
    """mode coco91: COCO category id -> 0..79 (None for unknown ids); contiguous: unchanged."""
    return TO_CONTIGUOUS.get(int(label)) if mode == "coco91" else int(label)


def guess_label_mode(labels, scores):
    """Traffic video: car is 2 (contiguous) or 3 (COCO id); the other id is a rare class (motorcycle / bicycle).
    The most frequent class alone misread a first batch of a sunny clip (Oct 22)."""
    confident = labels[scores >= .5].astype(int)
    cars_contiguous, cars_coco = int((confident == 2).sum()), int((confident == 3).sum())
    return "contiguous" if cars_contiguous > cars_coco else "coco91"


def lwdetr_run(session, frames, confidence, top_k=300):
    """-> per frame [(contiguous label, score, l, t, r, b normalized)] like lwdetr_decode."""
    out = []
    for frame in frames:
        boxes, logits = session.run(["dets", "labels"], {"input": preprocess([frame], "imagenet")})
        if boxes.ndim == 4:  # some exports keep decoder layers
            boxes, logits = boxes[-1], logits[-1]
        flat = (1 / (1 + np.exp(-np.clip(logits[0].astype(np.float64), -80, 80)))).reshape(-1)
        order = np.argsort(-flat, kind="stable")[:top_k]
        rows = []
        for idx in order:
            if flat[idx] < confidence:
                break
            query, category = divmod(int(idx), 91)
            label = TO_CONTIGUOUS.get(category)
            if label is None:
                continue
            cx, cy, w, h = (float(v) for v in boxes[0, query])
            rows.append([label, round(float(flat[idx]), 5), max(0.0, cx - w / 2), max(0.0, cy - h / 2),
                         min(1.0, cx + w / 2), min(1.0, cy + h / 2)])
        out.append(rows)
    return out


def cloud_frame(frame, size, quality):
    """v50 Cloud input: the Edge's JPEG (quality 1..100) or its square resize alone (quality 0)."""
    import cv2
    if not quality:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from edge_agent.control_v1 import encode_cloud_jpeg
    return cv2.imdecode(np.frombuffer(encode_cloud_jpeg(frame, size, quality), np.uint8), cv2.IMREAD_COLOR)


class GpuGuard:
    """Keep the GPU under a power / temperature limit by pacing batches (249's 4090 fell off the bus twice
    on Oct 22 under ~300 W sustained load). Stops the run when nvidia-smi can no longer reach the GPU."""

    def __init__(self, power_w, temp_c, every=10, query=None):
        self.power_w, self.temp_c, self.every = power_w, temp_c, every
        self.query = query or self._nvidia_smi
        self.delay, self.calls, self.slept = 0.0, 0, 0.0

    @staticmethod
    def _nvidia_smi():
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        power, temp = (float(x) for x in out.stdout.splitlines()[0].split(","))  # raises if the GPU is gone
        return power, temp

    def after_batch(self):
        self.calls += 1
        if self.calls % self.every == 0:
            try:
                power, temp = self.query()
            except Exception as exc:
                raise SystemExit(f"GPU unreachable, stopping: {exc}")
            if power > self.power_w or temp >= self.temp_c:
                self.delay = min(self.delay + .05, 1.0)
                time.sleep(2)
                self.slept += 2
            elif power < self.power_w - 40:
                self.delay = max(self.delay - .01, 0.0)
        if self.delay:
            time.sleep(self.delay)
            self.slept += self.delay


def dump_video(session, video, out, *, crop_bottom, batch, confidence, labels_mode, normalize, limit,
               jpeg_quality=None, jpeg_size=640, input_size=640, guard=None):
    import cv2
    capture = cv2.VideoCapture(str(video))
    frames, rows, index, mode, started = [], [], 0, labels_mode, time.perf_counter()

    def flush():
        nonlocal mode
        if mode == "lwdetr":
            for k, boxes in enumerate(lwdetr_run(session, frames, confidence)):
                rows.append({"frame": index - len(frames) + k, "boxes": boxes})
            frames.clear()
            return
        h, w = frames[0].shape[:2]
        sizes = np.array([[w, h]] * len(frames), dtype=np.int64)
        labels, boxes, scores = session.run(["labels", "boxes", "scores"],
                                            {"images": preprocess(frames, normalize, input_size),
                                             "orig_target_sizes": sizes})
        if mode == "auto":
            mode = guess_label_mode(labels, scores)
            print(f"[dump] label mode -> {mode}", flush=True)
        for k in range(len(frames)):
            keep = scores[k] >= confidence
            out_boxes = []
            for lab, box, score in zip(labels[k][keep], boxes[k][keep], scores[k][keep]):
                cls = to_contiguous(lab, mode)
                if cls is None:
                    continue
                l, t, r, b = (float(v) for v in box)
                out_boxes.append([cls, round(float(score), 5), max(0.0, l / w), max(0.0, t / h),
                                  min(1.0, r / w), min(1.0, b / h)])
            rows.append({"frame": index - len(frames) + k, "boxes": out_boxes})
        frames.clear()

    while limit is None or index < limit:
        ok, frame = capture.read()
        if not ok:
            break
        if crop_bottom:
            frame = frame[:-crop_bottom]
        if jpeg_quality is not None:
            # Normalized boxes are unchanged by the square stretch, so dumps stay comparable.
            frame = cloud_frame(frame, jpeg_size, jpeg_quality)
        frames.append(frame)
        index += 1
        if len(frames) == batch:
            flush()
            if guard is not None:
                guard.after_batch()
    if frames:
        flush()
    capture.release()
    if not rows:
        raise SystemExit(f"no frames decoded from {video}")
    part = out.with_suffix(".json.part")
    part.write_text(json.dumps({"artifact": str(session._model_path), "runtime": session.get_providers()[0],
                                "label_mode": mode, "normalize": normalize, "input_size": input_size,
                                "jpeg_quality": jpeg_quality,
                                "jpeg_size": jpeg_size if jpeg_quality is not None else None, "frames": rows}))
    part.replace(out)
    return len(rows), time.perf_counter() - started, mode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx")
    p.add_argument("video_dir")
    p.add_argument("out_dir")
    p.add_argument("--crop-bottom", type=int, default=64, help="identity band of live-transcoded videos")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--confidence", type=float, default=0.05, help="keep low for mAP; F1 is rescored")
    p.add_argument("--labels", choices=("auto", "coco91", "contiguous"), default="auto")
    p.add_argument("--normalize", choices=("none", "imagenet"), default="none")
    p.add_argument("--limit-frames", type=int, default=None, help="smoke test: frames per video")
    p.add_argument("--only", default="", help="comma separated video stems")
    p.add_argument("--cpu", action="store_true", help="allow CPU (smoke tests only)")
    p.add_argument("--jpeg-quality", type=int, choices=range(0, 101), metavar="0..100", default=None,
                   help="v50 Cloud input: Edge JPEG at this quality (0 = square resize only)")
    p.add_argument("--jpeg-size", type=int, default=640)
    p.add_argument("--input-size", type=int, default=640, help="square model input of a fixed-size export (e.g. 320)")
    p.add_argument("--power-limit-w", type=float, default=0, help="pace batches to stay under this GPU power (0 = off)")
    p.add_argument("--max-temp-c", type=float, default=80)
    args = p.parse_args(argv)
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("CUDAExecutionProvider required (use --cpu only for smoke tests)")
    session = ort.InferenceSession(args.onnx, providers=providers if not args.cpu else ["CPUExecutionProvider"])
    session._model_path = args.onnx
    names = {i.name for i in session.get_inputs()}
    if names == {"input"}:
        mode = "lwdetr"
    elif names == {"images", "orig_target_sizes"}:
        mode = args.labels
    else:
        raise SystemExit(f"unexpected ONNX inputs {names}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {x for x in args.only.split(",") if x}
    videos = [v for v in sorted(Path(args.video_dir).glob("*.mp4")) if not only or v.stem in only]
    guard = GpuGuard(args.power_limit_w, args.max_temp_c) if args.power_limit_w else None
    for video in videos:
        out = out_dir / f"{video.stem}.json"
        if out.exists() and out.stat().st_size:
            continue
        n, seconds, mode = dump_video(session, video, out, crop_bottom=args.crop_bottom, batch=args.batch,
                                      confidence=args.confidence, labels_mode=mode, normalize=args.normalize,
                                      limit=args.limit_frames, jpeg_quality=args.jpeg_quality,
                                      jpeg_size=args.jpeg_size, input_size=args.input_size, guard=guard)
        paced = f" paced_s={guard.slept:.0f}" if guard else ""
        print(f"DUMP {video.stem} frames={n} {n / seconds:.1f} fps{paced} -> {out}", flush=True)


if __name__ == "__main__":
    main()
