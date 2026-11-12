"""UA-DETRAC (official layout) -> videos, and per-second detection quality of a model dump.

  index: python -B scripts/detrac.py index ROOT                       # sequences, frames, weather, split
  video: python -B scripts/detrac.py video ROOT OUT_DIR [--only MVI_20011,...]
  score: python -B scripts/detrac.py score GT.xml DUMP.json --out quality.jsonl [--window-s 1]
  scene: python -B scripts/detrac.py scene VIDEO.mp4|VIDEO_DIR --out scene.jsonl  # pixel features per second
  table: python -B scripts/detrac.py table ROOT DUMP_DIR --model yolov8n --out yolov8n.jsonl [--min-conf 0.5]
  playlist: python -B scripts/detrac.py playlist ROOT VIDEO_DIR clip.mp4 [--split train --minutes 30]
  windows:  python -B scripts/detrac.py windows state.jsonl clip.json --out labeled.jsonl  # window -> scene

Layout searched under ROOT (case-insensitive folder names):
  images:      <seq>/img00001.jpg ...   (e.g. Insight-MVT_Annotation_Train/MVI_20011/)
  annotations: <seq>.xml or <seq>_v3.xml (DETRAC-Train-Annotations-XML, DETRAC-Test-Annotations-XML)
Split comes from the folder names (train/test). DUMP.json is `edge_parity.py dump --stride 1` output:
normalized [class_id, conf, left, top, right, bottom] per decoded frame index.

Scoring protocol (vehicle detection, per window of `window_s` seconds of source video):
  * UA-DETRAC labels only vehicles, so COCO car/bus/truck predictions form one "vehicle" class;
    other predicted classes (person, motorcycle, ...) are ignored, never counted as false positives.
  * Ignored regions: ground truth and predictions whose box center lies inside one are dropped.
  * Greedy match by IoU >= 0.5 per frame, highest confidence first. Window F1 = 2TP/(2TP+FP+FN);
    a window with neither ground truth nor predictions scores 1.0 (nothing to find, nothing invented).
  * Frames absent from the XML are unlabeled (never "empty": every listed frame has a vehicle) and are
    not scored; a window without any labeled frame yields no row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

VEHICLE_CLASSES = frozenset({2, 5, 7})  # COCO car, bus, truck
DETRAC_FPS = 25
ANALYTICS_FPS = 15  # live analytics / shown preview FPS
WIDTH, HEIGHT = 960, 540


def parse_annotation(path) -> dict:
    """-> {name, weather, camera, ignored:[(l,t,r,b)], frames:{num: [(l,t,r,b)]}} in pixels."""
    root = ET.parse(path).getroot()
    attrs = root.find("sequence_attribute")
    ignored = [_box(b) for b in root.iterfind("ignored_region/box")]
    frames = {}
    for frame in root.iterfind("frame"):
        frames[int(frame.get("num"))] = [_box(t.find("box")) for t in frame.iterfind("target_list/target")]
    return {"name": root.get("name"), "weather": None if attrs is None else attrs.get("sence_weather"),
            "camera": None if attrs is None else attrs.get("camera_state"), "ignored": ignored, "frames": frames}


def _box(node):
    left, top = float(node.get("left")), float(node.get("top"))
    return (left, top, left + float(node.get("width")), top + float(node.get("height")))


def _annotations(root) -> dict:
    xmls = {}
    for xml in Path(root).rglob("*.xml"):
        xmls.setdefault(re.sub(r"(_v\d+)?$", "", xml.stem), xml)
    return xmls


def _split(path) -> str | None:
    lowered = str(path).lower()
    return "test" if "test" in lowered else "train" if "train" in lowered else None


def index(root) -> list[dict]:
    root = Path(root)
    xmls = _annotations(root)
    out = []
    for seq in sorted({p.parent for p in root.rglob("img00001.jpg")}):
        xml = xmls.get(seq.name)
        # Kaggle v2 keeps all images in one DETRAC-Images folder; then the annotation folder decides.
        lowered = str(seq.relative_to(root)).lower()
        if "train" not in lowered and "test" not in lowered and xml is not None:
            lowered = str(xml.relative_to(root)).lower()
        info = parse_annotation(xml) if xml else {}
        out.append({"sequence": seq.name, "images": str(seq), "frames": len(list(seq.glob("img*.jpg"))),
                    "annotation": str(xml) if xml else None, "weather": info.get("weather"),
                    "camera": info.get("camera"), "split": _split(lowered)})
    return out


def table(root, dump_dir, model, *, min_conf=0.0, window_s=1.0, analyze_fps=None, display_fps=ANALYTICS_FPS):
    """Score every <sequence>.json of one model's dump folder -> rows tagged with model and split.

    min_conf > the live threshold re-scores all models at one common threshold (dumps keep confidences);
    it cannot go below the threshold the dump was made with.
    """
    xmls = _annotations(root)
    for path in sorted(Path(dump_dir).glob("*.json")):
        xml = xmls.get(path.stem)
        if xml is None:
            print(f"[table] skip {path.name}: no annotation", file=sys.stderr)
            continue
        dump = json.loads(path.read_text(encoding="utf-8"))
        if min_conf:
            dump["frames"] = [{**f, "boxes": [b for b in f["boxes"] if b[1] >= min_conf]} for f in dump["frames"]]
        split = _split(xml.relative_to(root))
        for row in score(xml, dump, window_s=window_s, analyze_fps=analyze_fps, display_fps=display_fps):
            yield {**row, "model": model, "split": split, **({"fps": analyze_fps} if analyze_fps else {})}


def video(root, out_dir, only=None, ffmpeg="ffmpeg", crf=18) -> list[Path]:
    """High-quality H.264 (CRF 18, about 1/4 the size of CRF 12) decoded identically by every model host."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for row in index(root):
        if only and row["sequence"] not in only:
            continue
        target = out_dir / f"{row['sequence']}.mp4"
        if target.exists():
            made.append(target)
            continue
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-framerate", str(DETRAC_FPS),
                        "-i", str(Path(row["images"]) / "img%05d.jpg"), "-c:v", "libx264", "-crf", str(crf),
                        "-preset", "medium", "-pix_fmt", "yuv420p", str(target.with_suffix(".tmp.mp4"))], check=True)
        target.with_suffix(".tmp.mp4").replace(target)
        made.append(target)
    return made


def playlist(root, video_dir, out, *, split="train", minutes=30.0, seed=0, ffmpeg="ffmpeg") -> list[dict]:
    """Concatenate whole sequences (weather round-robin, seeded order) into one clip for live control runs.

    Stream copy (no re-encode), so pixels equal the per-sequence videos the quality table was made from.
    Writes OUT and OUT.json manifest [{sequence, weather, start_frame, frames}] (0-based clip frames).
    """
    import random
    rng = random.Random(seed)
    by_weather = {}
    for row in index(root):
        if row["split"] == split and (Path(video_dir) / f"{row['sequence']}.mp4").exists():
            by_weather.setdefault(row["weather"], []).append(row)
    for rows in by_weather.values():
        rng.shuffle(rows)
    budget, chosen, start = round(minutes * 60 * DETRAC_FPS), [], 0
    while start < budget and any(by_weather.values()):
        for weather in sorted(by_weather, key=str):
            if by_weather[weather] and start < budget:
                row = by_weather[weather].pop()
                chosen.append({"sequence": row["sequence"], "weather": weather, "start_frame": start,
                               "frames": row["frames"]})
                start += row["frames"]
    if not chosen:
        raise SystemExit(f"no {split} videos found in {video_dir}")
    out = Path(out)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{(Path(video_dir) / (c['sequence'] + '.mp4')).resolve().as_posix()}'\n"
                               for c in chosen), encoding="utf-8")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(out)], check=True)
    listing.unlink()
    out.with_suffix(".json").write_text(json.dumps({"fps": DETRAC_FPS, "total_frames": start, "split": split,
                                                    "seed": seed, "sequences": chosen}, indent=1), encoding="utf-8")
    return chosen


def locate(frame_id, manifest):
    """Source frame id of a looping playlist run -> (sequence, 0-based frame in it, second in it)."""
    position = frame_id % manifest["total_frames"]
    for item in manifest["sequences"]:
        if position < item["start_frame"] + item["frames"]:
            local = position - item["start_frame"]
            return item["sequence"], local, local // manifest["fps"]
    raise ValueError("manifest frames do not cover the clip")


def label_windows(state_lines, manifest, start_frame=0):
    """Tag each control-v1 window with the scene it showed: the sequence/second of its median frame.
    start_frame: where a staged run (CONTROL_START_S) began in the clip; frame IDs count from there."""
    for line in state_lines:
        window = json.loads(line)
        ids = sorted(f["frame_id"] for f in window.get("frames", ()))
        if ids:
            sequence, _frame, second = locate(start_frame + ids[len(ids) // 2], manifest)
            window["scene"] = {"sequence": sequence, "second": second, "frames": len(ids)}
        yield window


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ignored(box, regions):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions)


def match_frame(gt, predictions, ignored=(), threshold=.5):
    """gt: [(l,t,r,b)] px; predictions: [(conf, (l,t,r,b))] px vehicles -> (tp, fp, fn)."""
    gt = [g for g in gt if not _ignored(g, ignored)]
    predictions = [p for p in predictions if not _ignored(p[1], ignored)]
    used, tp = set(), 0
    for _conf, box in sorted(predictions, key=lambda p: -p[0]):
        best, j = threshold, None
        for k, g in enumerate(gt):
            if k not in used and (v := iou(box, g)) >= best:
                best, j = v, k
        if j is not None:
            used.add(j)
            tp += 1
    return tp, len(predictions) - tp, len(gt) - tp


def average_precision(root, dump_dir, *, split="test", thresholds=tuple(x / 100 for x in range(50, 100, 5)),
                      width=WIDTH, height=HEIGHT):
    """Class-agnostic vehicle AP over every annotated frame of `split` (same vehicle set and ignored regions
    as the F1 score) -> {"AP50", "AP75", "AP50:95", "sequences", "frames", "gt"}. COCO 101-point AP.
    Dumps keep only boxes above their dump threshold, so AP is a lower bound for high-threshold dumps."""
    import numpy as np
    xmls = _annotations(root)
    hits = {t: [] for t in thresholds}  # (confidence, is_true_positive)
    gt_total = frames_n = sequences = 0
    for path in sorted(Path(dump_dir).glob("*.json")):
        xml = xmls.get(path.stem)
        if xml is None or _split(xml.relative_to(root)) != split:
            continue
        ann = parse_annotation(xml)
        sequences += 1
        for item in json.loads(path.read_text(encoding="utf-8"))["frames"]:
            num = item["frame"] + 1
            if num not in ann["frames"]:
                continue
            frames_n += 1
            gt = [g for g in ann["frames"][num] if not _ignored(g, ann["ignored"])]
            preds = sorted(((b[1], (b[2] * width, b[3] * height, b[4] * width, b[5] * height))
                            for b in item["boxes"] if int(b[0]) in VEHICLE_CLASSES), key=lambda p: -p[0])
            preds = [p for p in preds if not _ignored(p[1], ann["ignored"])]
            gt_total += len(gt)
            if not preds:
                continue
            m = np.array([[iou(box, g) for g in gt] for _c, box in preds]) if gt else np.zeros((len(preds), 0))
            for t in thresholds:
                used = set()
                for k, (conf, _box) in enumerate(preds):
                    j, best = None, t
                    for g in range(m.shape[1]):
                        if g not in used and m[k, g] >= best:
                            j, best = g, m[k, g]
                    if j is not None:
                        used.add(j)
                    hits[t].append((conf, j is not None))
    ap = {}
    for t, rows in hits.items():
        rows.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in rows]) if rows else np.zeros(0)
        precision = tp / np.arange(1, len(rows) + 1) if rows else np.zeros(0)
        recall = tp / max(1, gt_total) if rows else np.zeros(0)
        # COCO: precision envelope sampled at 101 recall points
        envelope = np.maximum.accumulate(precision[::-1])[::-1] if rows else precision
        points = [envelope[np.searchsorted(recall, r, side="left")] if rows and r <= recall[-1] else 0.0
                  for r in np.linspace(0, 1, 101)]
        ap[t] = float(np.mean(points))
    return {"AP50": ap.get(.5), "AP75": ap.get(.75), "AP50:95": float(np.mean(list(ap.values()))),
            "sequences": sequences, "frames": frames_n, "gt": gt_total}


def _on_grid(index, rate, source):
    """Frame `index` of a `source` FPS sequence is kept when thinning it to `rate` on an even grid (live cadence)."""
    return index == 0 or (index * rate) // source != ((index - 1) * rate) // source


def score(annotation, dump, *, window_s=1.0, fps=DETRAC_FPS, width=WIDTH, height=HEIGHT, analyze_fps=None,
          display_fps=ANALYTICS_FPS):
    """-> one row per window: {sequence, window, start_frame, frames, tp, fp, fn, gt, f1, weather}.

    analyze_fps: score what the live page shows at display_fps when only analyze_fps of those frames are
    analyzed and every other shown frame keeps the boxes of the latest analyzed one (RAW overlay).
    """
    ann = parse_annotation(annotation) if not isinstance(annotation, dict) else annotation
    frames = dump["frames"]
    if analyze_fps is not None:
        shown, held, j = [], None, 0
        for item in sorted(frames, key=lambda f: f["frame"]):
            if not _on_grid(item["frame"], display_fps, fps):
                continue
            if held is None or _on_grid(j, analyze_fps, display_fps):
                held = item["boxes"]
            shown.append({**item, "boxes": held})
            j += 1
        frames = shown
    # Dump frame index is 0-based over decoded video; DETRAC frame numbers start at 1.
    per_window, size = {}, max(1, round(window_s * fps))
    for item in frames:
        num = item["frame"] + 1
        # 15 sequences have contiguous runs absent from the XML (1.1% of frames) while no listed frame
        # is ever empty: absent means unlabeled, not vehicle-free, so it is not scored at all.
        if num not in ann["frames"]:
            continue
        preds = [(b[1], (b[2] * width, b[3] * height, b[4] * width, b[5] * height))
                 for b in item["boxes"] if int(b[0]) in VEHICLE_CLASSES]
        tp, fp, fn = match_frame(ann["frames"].get(num, []), preds, ann["ignored"])
        row = per_window.setdefault(item["frame"] // size, [0, 0, 0, 0, 0])
        row[0] += tp; row[1] += fp; row[2] += fn; row[3] += 1; row[4] += tp + fn
    observed = observed_features(frames, size=size, fps=fps)
    out = []
    for window, (tp, fp, fn, count, gt) in sorted(per_window.items()):
        denominator = 2 * tp + fp + fn
        out.append({"sequence": ann["name"], "window": window, "start_frame": window * size + 1,
                    "frames": count, "tp": tp, "fp": fp, "fn": fn, "gt": gt,
                    "f1": 1.0 if denominator == 0 else 2 * tp / denominator, "weather": ann["weather"],
                    **observed.get(window, {})})
    return out


LIVE_VEHICLES = frozenset({1, 2, 3, 5, 7})  # same set as edge_agent.resources_v2.VEHICLE_CLASSES
SMALL_AREA = (32 / 1280) * (32 / 720)


def _box_change(previous, current, threshold=.3):
    """Live formula of edge_agent.phase7._box_change with IoU association standing in for track IDs.

    shortcut: greedy IoU >= 0.3 matching approximates ByteTrack identities; replay the dumps through
    the Cloud tracker if the offline/live box_change distributions diverge.
    """
    if not previous and not current:
        return 0.0
    pairs, used = [], set()
    for a in previous:
        best, j = threshold, None
        for k, b in enumerate(current):
            if k not in used and (v := iou(a, b)) >= best:
                best, j = v, k
        if j is not None:
            used.add(j)
            pairs.append(best)
    union = len(previous) + len(current) - len(pairs)
    changed = (union - len(pairs)) / max(1, union)
    geometry = 1.0 - sum(pairs) / len(pairs) if pairs else 1.0
    return min(1.0, max(0.0, .5 * changed + .5 * geometry))


def observed_features(frames, *, size, fps=DETRAC_FPS):
    """Label-free features the live Edge would compute from this model's own results, per window.

    Only frames on the 15 FPS analytics cadence count (the live controller sees those results).
    """
    windows, previous = {}, None
    for item in sorted(frames, key=lambda f: f["frame"]):
        i = item["frame"]
        if i and int(i * ANALYTICS_FPS / fps) == int((i - 1) * ANALYTICS_FPS / fps):
            continue
        boxes = [tuple(b[2:6]) for b in item["boxes"] if int(b[0]) in LIVE_VEHICLES]
        row = windows.setdefault(i // size, {"count": [], "small": [], "change": []})
        row["count"].append(len(boxes))
        if boxes:
            row["small"].append(sum((r - l) * (b - t) < SMALL_AREA for l, t, r, b in boxes) / len(boxes))
        if previous is not None:
            row["change"].append(_box_change(previous, boxes))
        previous = boxes
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {w: {"vehicle_count": mean(r["count"]), "small_ratio": mean(r["small"]), "box_change": mean(r["change"])}
            for w, r in windows.items()}


def scene(video_path, *, window_s=1.0, sample_ms=200):
    """Pixel features per window exactly as the live capture loop samples them (every 200 ms)."""
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from edge_agent.resources_v2 import scene_features
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or DETRAC_FPS
    size, step = max(1, round(window_s * fps)), max(1, round(sample_ms / 1000 * fps))
    windows, previous, index = {}, None, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            features, previous = scene_features(frame, previous)
            row = windows.setdefault(index // size, {})
            for name, value in features.items():
                row.setdefault(name, []).append(value)
        index += 1
    capture.release()
    return [{"window": w, **{k: sum(v) / len(v) for k, v in row.items()}} for w, row in sorted(windows.items())]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").add_argument("root")
    v = sub.add_parser("video")
    v.add_argument("root"); v.add_argument("out_dir")
    v.add_argument("--only", default="", help="comma separated sequence names")
    v.add_argument("--ffmpeg", default="ffmpeg")
    v.add_argument("--crf", type=int, default=18)
    s = sub.add_parser("score")
    s.add_argument("annotation"); s.add_argument("dump")
    s.add_argument("--out", required=True)
    s.add_argument("--window-s", type=float, default=1.0)
    c = sub.add_parser("scene")
    c.add_argument("video", help="a video, or a folder of .mp4"); c.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("root"); t.add_argument("dump_dir")
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--min-conf", type=float, default=0.0)
    pl = sub.add_parser("playlist")
    pl.add_argument("root"); pl.add_argument("video_dir"); pl.add_argument("out")
    pl.add_argument("--split", default="train")
    pl.add_argument("--minutes", type=float, default=30.0)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--ffmpeg", default="ffmpeg")
    a = sub.add_parser("ap")
    a.add_argument("root"); a.add_argument("dump_dir")
    a.add_argument("--split", default="test")
    w = sub.add_parser("windows")
    w.add_argument("state"); w.add_argument("manifest"); w.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "index":
        rows = index(args.root)
        for row in rows:
            print(json.dumps(row))
        by = {}
        for row in rows:
            key = (row["split"], row["weather"])
            by[key] = by.get(key, 0) + row["frames"]
        print(json.dumps({"sequences": len(rows), "annotated": sum(r["annotation"] is not None for r in rows),
                          "frames_by_split_weather": {f"{k[0]}/{k[1]}": n for k, n in sorted(by.items(), key=str)}}))
    elif args.cmd == "video":
        made = video(args.root, args.out_dir, {x for x in args.only.split(",") if x}, args.ffmpeg, args.crf)
        print(f"VIDEOS {len(made)} -> {args.out_dir}")
    elif args.cmd == "scene":
        source = Path(args.video)
        videos = sorted(source.glob("*.mp4")) if source.is_dir() else [source]
        count = 0
        with open(args.out, "w", encoding="utf-8") as out:
            for video_path in videos:
                for row in scene(video_path):
                    out.write(json.dumps({"sequence": video_path.stem, **row}) + "\n")
                    count += 1
        print(f"SCENE videos={len(videos)} windows={count} -> {args.out}")
    elif args.cmd == "playlist":
        chosen = playlist(args.root, args.video_dir, args.out, split=args.split, minutes=args.minutes,
                          seed=args.seed, ffmpeg=args.ffmpeg)
        weathers = {}
        for c in chosen:
            weathers[c["weather"]] = weathers.get(c["weather"], 0) + c["frames"]
        print(f"PLAYLIST sequences={len(chosen)} frames={sum(c['frames'] for c in chosen)} "
              f"by_weather={weathers} -> {args.out}")
    elif args.cmd == "ap":
        print(json.dumps(average_precision(args.root, args.dump_dir, split=args.split)))
    elif args.cmd == "windows":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        count = 0
        marker = Path(args.state).with_name("start_frame")  # written by control.sh next to state.jsonl
        start = int(marker.read_text()) if marker.is_file() else 0
        with open(args.state, encoding="utf-8") as lines, open(args.out, "w", encoding="utf-8") as out:
            for window in label_windows(lines, manifest, start):
                out.write(json.dumps(window) + "\n")
                count += 1
        print(f"WINDOWS {count} labeled (start_frame={start}) -> {args.out}")
    elif args.cmd == "table":
        rows = 0
        f1 = {}
        with open(args.out, "w", encoding="utf-8") as out:
            for row in table(args.root, args.dump_dir, args.model, min_conf=args.min_conf):
                out.write(json.dumps(row) + "\n")
                rows += 1
                f1.setdefault(row["split"], []).append(row["f1"])
        means = {k: round(sum(v) / len(v), 4) for k, v in f1.items()}
        print(f"TABLE {args.model} windows={rows} mean_f1_by_split={means} -> {args.out}")
    else:
        dump = json.loads(Path(args.dump).read_text(encoding="utf-8"))
        rows = score(args.annotation, dump, window_s=args.window_s)
        with open(args.out, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps({**row, "model_artifact": dump.get("artifact")}) + "\n")
        mean = sum(r["f1"] for r in rows) / len(rows) if rows else None
        print(f"SCORED windows={len(rows)} mean_f1={mean:.4f} -> {args.out}" if rows else "SCORED none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
