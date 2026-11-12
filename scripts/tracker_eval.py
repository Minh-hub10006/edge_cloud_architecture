"""Are track IDs stable? (8/11: 60 s of VN gave 624-3152 new track IDs for 6-34 objects.) Detections once, trackers many.

  dump: python -B scripts/tracker_eval.py dump OUT.json MODEL.onnx VIDEO.mp4 [VIDEO ...] [--seconds 40]
  eval: python -B scripts/tracker_eval.py eval OUT.json [--xml-dir DETRAC-Test-Annotations-XML]

dump: RT-DETRv2-S (full ONNX, CPU, INTER_AREA 640 like the Cloud) at 15 fps (frames on the 15 fps grid of the source),
      every box with score >= 0.1, normalized.
eval: cloud_core.tracking.CloudByteTrack (the tracker the Cloud and mode A use) replayed under several settings.
  DETRAC videos (with <name>.xml ground truth target IDs; vehicles = car/bus/truck predictions, ignored regions dropped,
  IoU >= .5 greedy): per ground-truth vehicle, how often its matched track ID changes (ID switches), how many IDs it
  gets, and ID recall (frames matched to its most frequent ID / its frames).
  Other videos (VN, no IDs): track IDs created per minute per object present, and median track life.
"""
import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VEHICLES = {2, 5, 7}
CONFIGS = {   # name: (minimum score sent to the tracker, CloudByteTrack options)
    "current (Cloud 0.5 cut, match .5)": (.5, {}),
    "match .8": (.5, {"match_thresh": .8}),
    "match .8, buffer 30": (.5, {"match_thresh": .8, "track_buffer": 30}),
    "match .8, low boxes to tracker (cut .1)": (.1, {"match_thresh": .8}),
    "match .7, no score fusion": (.5, {"match_thresh": .7, "fuse_score": False}),
    "match .9": (.5, {"match_thresh": .9}),
    "match .8, new track .6": (.5, {"match_thresh": .8, "new_track_thresh": .6}),
    "match .8, high .5, new track .6": (.5, {"match_thresh": .8, "track_high_thresh": .5, "new_track_thresh": .6}),
    "match .8, buffer 90": (.5, {"match_thresh": .8, "track_buffer": 90}),
    "match .8, new track .6, low boxes (cut .1)": (.1, {"match_thresh": .8, "new_track_thresh": .6}),
    "match .8, high .5, new .6, low boxes (cut .1)": (.1, {"match_thresh": .8, "track_high_thresh": .5, "new_track_thresh": .6}),
    "ultralytics defaults": (.1, {"track_high_thresh": .25, "track_low_thresh": .1, "new_track_thresh": .25,
                                  "track_buffer": 30, "match_thresh": .8, "fuse_score": True}),
}


def dump(out, model, videos, seconds):
    import cv2
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    data = {}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, i, last = [], 0, -1
        while i < seconds * fps:
            ok, bgr = cap.read()
            if not ok:
                break
            slot = int(i * 15 / fps)
            if slot != last:   # the 15 fps analytics grid of this source
                last = slot
                rgb = cv2.cvtColor(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                x = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
                sc, bx, lb = sess.run(["scores", "boxes", "labels"], {"images": x, "orig_target_sizes": np.array([[w, h]])})
                boxes = [[int(l), round(float(c), 4), *(round(float(v), 5) for v in (b[0] / w, b[1] / h, b[2] / w, b[3] / h))]
                         for c, b, l in zip(sc[0], bx[0], lb[0]) if c >= .1]
                frames.append({"i": i, "pts_ns": round(i * 1e9 / fps), "boxes": boxes})
            i += 1
        data[Path(video).stem] = {"fps": fps, "width": w, "height": h, "frames": frames}
        print(f"{Path(video).stem}: {len(frames)} frames at 15 fps", flush=True)
    Path(out).write_text(json.dumps(data))


def track(seq, cut, options):
    from contracts import Box, DetectionResult, FrameIdentity, ModelMode
    from cloud_core.tracking import CloudByteTrack
    tracker = CloudByteTrack(frame_rate=15, **options)
    out = []
    for f in seq["frames"]:
        boxes = tuple(Box(max(0, min(1, l)), max(0, min(1, t)), max(0, min(1, r)), max(0, min(1, b)), c, k, str(k))
                      for k, c, l, t, r, b in f["boxes"] if c >= cut and r > l and b > t)
        ident = FrameIdentity("eval", "cam", "s", 1, f["i"], 1 + f["pts_ns"], f["pts_ns"])
        result = DetectionResult(frame=ident, model_mode=ModelMode.RTDETR_CLOUD, model_id="m", decision_id="d",
                                 boxes=boxes, inference_ms=1, produced_utc_ns=2 + f["pts_ns"], request_id=str(f["i"]),
                                 preview_pts=f["pts_ns"])
        tracked = tracker.update(result)
        out.append((f["i"], [b for b in tracked.boxes if b.track_id is not None and b.confidence >= max(cut, .5)]))
    return out


def ground_truth(xml):
    root = ET.parse(xml).getroot()
    box = lambda n: (float(n.get("left")), float(n.get("top")), float(n.get("left")) + float(n.get("width")),
                     float(n.get("top")) + float(n.get("height")))
    ignored = [box(b) for b in root.iterfind("ignored_region/box")]
    frames = {int(fr.get("num")) - 1: [(int(t.get("id")), box(t.find("box"))) for t in fr.iterfind("target_list/target")]
              for fr in root.iterfind("frame")}  # XML frames are 1-based
    return frames, ignored


def iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def id_metrics(tracked, seq, xml):
    frames, ignored = ground_truth(xml)
    inside = lambda b: any(r[0] <= (b[0] + b[2]) / 2 <= r[2] and r[1] <= (b[1] + b[3]) / 2 <= r[3] for r in ignored)
    w, h = seq["width"], seq["height"]
    history = defaultdict(list)   # gt id -> matched track id per labeled frame (None = missed)
    for i, boxes in tracked:
        if i not in frames:
            continue
        gt = [(g, b) for g, b in frames[i] if not inside(b)]
        preds = [(p.track_id, (p.left * w, p.top * h, p.right * w, p.bottom * h)) for p in boxes if p.class_id in VEHICLES]
        preds = [(t, b) for t, b in preds if not inside(b)]
        pairs = sorted(((iou(gb, pb), g, t) for g, gb in gt for t, pb in preds), reverse=True)
        used_g, used_t = set(), set()
        for v, g, t in pairs:
            if v < .5:
                break
            if g not in used_g and t not in used_t:
                used_g.add(g)
                used_t.add(t)
                history[g].append(t)
        for g, _ in gt:
            if g not in used_g:
                history[g].append(None)
    switches = ids = id_tp = total = 0
    for seq_ids in history.values():
        matched = [t for t in seq_ids if t is not None]
        total += len(seq_ids)
        if matched:
            switches += sum(a != b for a, b in zip(matched, matched[1:]))
            ids += len(set(matched))
            id_tp += Counter(matched).most_common(1)[0][1]
    n = len(history)
    return {"gt_vehicles": n, "id_switches": switches, "ids_per_vehicle": round(ids / max(n, 1), 2),
            "id_recall": round(id_tp / max(total, 1), 3)}


def life_metrics(tracked):
    seen = defaultdict(int)
    objects = 0
    for _, boxes in tracked:
        objects += len(boxes)
        for b in boxes:
            seen[b.track_id] += 1
    minutes = len(tracked) / 15 / 60
    mean_objects = objects / max(len(tracked), 1)
    return {"objects_per_frame": round(mean_objects, 1), "ids_per_object_minute": round(len(seen) / max(mean_objects * minutes, 1e-9), 1),
            "median_life_frames": statistics.median(seen.values()) if seen else 0}


def evaluate(data_path, xml_dir):
    data = json.loads(Path(data_path).read_text())
    for name, seq in data.items():
        xml = Path(xml_dir) / f"{name}.xml" if xml_dir else None
        print(f"== {name} ({len(seq['frames'])} frames){' with ground-truth IDs' if xml and xml.exists() else ''}")
        for label, (cut, options) in CONFIGS.items():
            tracked = track(seq, cut, options)
            m = id_metrics(tracked, seq, xml) if xml and xml.exists() else life_metrics(tracked)
            print(f"   {label:42s} {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "eval"])
    ap.add_argument("data")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--xml-dir", default="")
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.data, a.rest[0], a.rest[1:], a.seconds)
    else:
        evaluate(a.data, a.xml_dir)


if __name__ == "__main__":
    main()
