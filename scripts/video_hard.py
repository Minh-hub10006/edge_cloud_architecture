"""UA-DETRAC videos with many hard scenes (docs/PLAN_VIDEO_HARD_VI.md).

  python -B scripts/video_hard.py table ANNOTATION_ROOT [--out results/detrac/hardness.jsonl]
  python -B scripts/video_hard.py segments VIDEO_DIR [--table ...hardness.jsonl] [--out results/detrac/segments.jsonl]
  python -B scripts/video_hard.py video train|switch VIDEO_DIR OUT.mp4 [--segments ...segments_int8.jsonl] [--seed 0]

table (V1): per-second F1 at 15 FPS of D-FINE-X, RT-DETRv2 and YOLO11s from the existing dumps, at the
MODEL_COMPARE.md thresholds, and the gaps gE = DX - Y11 (Cloud vs Edge), gC = DX - RT (Cloud vs Cloud).
segments (V3): keyframe-aligned ~20 s pieces (stream copy keeps the scored pixels), labeled hard / easy / mid.
video (V4/V5): train = every hard + easy train segment; switch = test hard / easy blocks of 1-3 segments alternating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_compare  # noqa: E402
from detrac import DETRAC_FPS  # noqa: E402

DUMPS = {"dfine_x": ("results/detrac/cand_small/dfine_x", .65),
         "rtdetrv2": ("results/detrac/cand_small/rtdetrv2", .6),
         "yolo11s": ("results/detrac/quality/yolo11s", .25)}
HARD_E, HARD_C, EASY_E, EASY_C = .10, .05, .05, .03  # segment labels (mean gaps over its seconds)


def table(root, dumps=DUMPS):
    """-> [{split, sequence, second, weather, f1_<model>..., gE, gC}] for seconds scored by every model."""
    per = {}
    for name, (folder, threshold) in dumps.items():
        for split, pairs in model_compare.load(root, folder).items():
            for row in model_compare.f1_rows(pairs, threshold):
                key = (split, row["sequence"], row["window"])
                per.setdefault(key, {"split": split, "sequence": row["sequence"], "second": row["window"],
                                     "weather": row["weather"]})[f"f1_{name}"] = round(row["f1"], 4)
    rows = [r for r in per.values() if all(f"f1_{n}" in r for n in dumps)]
    for r in rows:
        r["gE"] = round(r["f1_dfine_x"] - r["f1_yolo11s"], 4)
        r["gC"] = round(r["f1_dfine_x"] - r["f1_rtdetrv2"], 4)
    return sorted(rows, key=lambda r: (r["split"], r["sequence"], r["second"]))


def keyframes(video, ffprobe="ffprobe"):
    """-> (keyframe frame numbers, total frames) of a 25 fps video."""
    import subprocess
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v", "-skip_frame", "nokey", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    keys = sorted({round(float(x.strip(", ")) * DETRAC_FPS) for x in out.stdout.split() if x.strip(", ")})
    total = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v", "-count_packets", "-show_entries",
                            "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, check=True)
    return keys, int(total.stdout.strip().strip(","))


def cut_points(keys, total, target_s=20):
    """Keyframe-aligned [start, end) frame ranges of ~target_s; a tail shorter than half is merged back."""
    target, bounds = target_s * DETRAC_FPS, [0]
    for k in keys[1:]:
        if k - bounds[-1] >= target:
            bounds.append(k)
    if total - bounds[-1] < target / 2 and len(bounds) > 1:
        bounds.pop()
    return list(zip(bounds, bounds[1:] + [total]))


def label(gE, gC):
    if gE >= HARD_E or gC >= HARD_C:
        return "hard"
    return "easy" if gE <= EASY_E and gC <= EASY_C else "mid"


def segments(rows, video_dir):
    """-> keyframe-aligned ~20 s segments with mean gaps over their scored seconds and a hard/easy/mid label."""
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["sequence"], []).append(r)
    out = []
    for sequence, seconds in sorted(by_seq.items()):
        keys, total = keyframes(Path(video_dir) / f"{sequence}.mp4")
        for start, end in cut_points(keys, total):
            inside = [r for r in seconds if start <= r["second"] * DETRAC_FPS < end]
            if not inside:
                continue
            gE = float(np.mean([r["gE"] for r in inside]))
            gC = float(np.mean([r["gC"] for r in inside]))
            out.append({"split": seconds[0]["split"], "sequence": sequence, "weather": seconds[0]["weather"],
                        "start_frame": start, "frames": end - start, "seconds_scored": len(inside),
                        "gE": round(gE, 4), "gC": round(gC, 4), "label": label(gE, gC),
                        **{f"f1_{n}": round(float(np.mean([r[f"f1_{n}"] for r in inside])), 4) for n in DUMPS}})
    return out


def _by_sequence(segs, rng):
    """Segments grouped by sequence (source order kept), sequence order shuffled."""
    groups = {}
    for s in segs:
        groups.setdefault(s["sequence"], []).append(s)
    names = sorted(groups)
    rng.shuffle(names)
    return [sorted(groups[n], key=lambda s: s["start_frame"]) for n in names]


def order_train(segs, seed=0):
    """V4: every hard and easy train segment (mid ones dropped), weather round-robin over shuffled sequences."""
    import random
    rng = random.Random(seed)
    by_weather = {}
    for group in _by_sequence([s for s in segs if s["split"] == "train" and s["label"] in ("hard", "easy")], rng):
        by_weather.setdefault(group[0]["weather"], []).append(group)
    out = []
    while any(by_weather.values()):
        for weather in sorted(by_weather, key=str):
            if by_weather[weather]:
                out += by_weather[weather].pop()
    return out


def order_switch(segs, seed=0):
    """V5: test hard and easy blocks alternate; a block is 1-3 segments (~20-60 s) so the period is irregular."""
    import random
    rng = random.Random(seed)
    pools = {lab: [s for g in _by_sequence([s for s in segs if s["split"] == "test" and s["label"] == lab], rng)
                   for s in g] for lab in ("hard", "easy")}
    out, lab = [], "easy"
    while pools["hard"] and pools["easy"]:
        lab = "hard" if lab == "easy" else "easy"
        for _ in range(rng.choice((1, 2, 3))):
            if pools[lab]:
                out.append(pools[lab].pop(0))
    return out


def _frames(path, ffprobe="ffprobe"):
    import subprocess
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().strip(","))


def build(chosen, video_dir, out, *, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    """Keyframe-aligned stream copy (pixels = the scored videos) + manifest. Each piece is cut on its own and its
    real frame count goes into the manifest: a copy cut keeps ~1 frame past the outpoint (Oct 23, 144 frames
    over a 38 min clip), which would shift every later label."""
    import subprocess
    import tempfile
    out = Path(out)
    items, start = [], 0
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        pieces = []
        for i, s in enumerate(chosen):
            piece, spec = Path(tmp) / f"{i:04d}.mp4", Path(tmp) / f"{i:04d}.txt"
            source = (Path(video_dir) / (s["sequence"] + ".mp4")).resolve().as_posix()
            spec.write_text("\n".join([f"file '{source}'", f"inpoint {s['start_frame'] / DETRAC_FPS:.2f}",
                                       f"outpoint {(s['start_frame'] + s['frames']) / DETRAC_FPS:.2f}", ""]),
                            encoding="utf-8")
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                            str(spec), "-c", "copy", str(piece)], check=True)
            frames = _frames(piece, ffprobe)
            items.append({"sequence": s["sequence"], "weather": s["weather"], "start_frame": start, "frames": frames,
                          "source_start_frame": s["start_frame"], "label": s["label"], "gE": s["gE"], "gC": s["gC"]})
            start += frames
            pieces.append(piece)
        listing = Path(tmp) / "all.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in pieces), encoding="utf-8")
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(listing), "-c", "copy", str(out)], check=True)
    counted = _frames(out, ffprobe)
    if counted != start:
        raise SystemExit(f"{out}: {counted} frames, manifest says {start}")
    switches = [it["start_frame"] for prev, it in zip(items, items[1:]) if it["label"] != prev["label"]]
    weight = np.array([it["frames"] for it in items], float)
    summary = {"minutes": round(start / DETRAC_FPS / 60, 1),
               "hard_share": round(float(weight[[it["label"] == "hard" for it in items]].sum() / weight.sum()), 3),
               "gE": round(float(np.average([it["gE"] for it in items], weights=weight)), 4),
               "gC": round(float(np.average([it["gC"] for it in items], weights=weight)), 4),
               "switches": len(switches)}
    out.with_suffix(".json").write_text(json.dumps({"fps": DETRAC_FPS, "total_frames": start, "summary": summary,
                                                    "switch_frames": switches, "sequences": items}, indent=1),
                                        encoding="utf-8")
    return summary


def check_pixels(clip, video_dir, manifest, pieces=4, frames=3):
    """The first pieces, decoded sequentially (OpenCV seeking in H.264 is not frame exact), must equal the source
    frames they claim. Returns the largest mean absolute pixel difference."""
    import cv2
    cap, index, worst = cv2.VideoCapture(str(clip)), 0, 0.0
    for it in manifest["sequences"][:pieces]:
        src = cv2.VideoCapture(str(Path(video_dir) / f"{it['sequence']}.mp4"))
        for _ in range(it["source_start_frame"]):
            src.read()
        while index < it["start_frame"]:
            cap.read()
            index += 1
        for _ in range(frames):
            (ok1, a), (ok2, b) = cap.read(), src.read()
            index += 1
            if not (ok1 and ok2):
                raise SystemExit(f"cannot read frame of {it['sequence']}")
            worst = max(worst, float(np.abs(a.astype(int) - b.astype(int)).mean()))
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table")
    t.add_argument("root")
    t.add_argument("--out", default="results/detrac/hardness.jsonl")
    g = sub.add_parser("segments")
    g.add_argument("video_dir")
    g.add_argument("--table", default="results/detrac/hardness.jsonl")
    g.add_argument("--out", default="results/detrac/segments.jsonl")
    v = sub.add_parser("video")
    v.add_argument("kind", choices=("train", "switch"))
    v.add_argument("video_dir")
    v.add_argument("out")
    v.add_argument("--segments", default="results/detrac/segments_int8.jsonl")
    v.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.command == "video":
        segs = [json.loads(x) for x in Path(args.segments).read_text(encoding="utf-8").splitlines()]
        chosen = (order_train if args.kind == "train" else order_switch)(segs, args.seed)
        summary = build(chosen, args.video_dir, args.out)
        manifest = json.loads(Path(args.out).with_suffix(".json").read_text(encoding="utf-8"))
        summary["pixel_diff_max"] = round(check_pixels(args.out, args.video_dir, manifest), 3)
        print(json.dumps(summary))
        return
    if args.command == "segments":
        rows = [json.loads(x) for x in Path(args.table).read_text(encoding="utf-8").splitlines()]
        segs = segments(rows, args.video_dir)
        Path(args.out).write_text("".join(json.dumps(s) + "\n" for s in segs), encoding="utf-8")
        for split in ("train", "test"):
            for lab in ("hard", "easy", "mid"):
                part = [s for s in segs if s["split"] == split and s["label"] == lab]
                print(json.dumps({"split": split, "label": lab, "segments": len(part),
                                  "minutes": round(sum(s["frames"] for s in part) / DETRAC_FPS / 60, 1),
                                  "gE": round(float(np.mean([s["gE"] for s in part])), 4) if part else None,
                                  "gC": round(float(np.mean([s["gC"] for s in part])), 4) if part else None}))
        print(f"-> {args.out}")
        return
    rows = table(args.root)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        f1 = {n: round(float(np.mean([r[f"f1_{n}"] for r in part])), 4) for n in DUMPS}
        print(json.dumps({"split": split, "sequences": len({r["sequence"] for r in part}), "seconds": len(part),
                          "mean_f1": f1, "gE": round(float(np.mean([r["gE"] for r in part])), 4),
                          "gC": round(float(np.mean([r["gC"] for r in part])), 4),
                          "hard_s": round(float(np.mean([r["gE"] >= .10 or r["gC"] >= .05 for r in part])), 3)}))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
