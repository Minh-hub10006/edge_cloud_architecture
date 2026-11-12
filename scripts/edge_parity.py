"""Cross-Edge parity: the same Edge model on two hosts must give the same boxes.

  dump:    python -B scripts/edge_parity.py dump --adapter A --artifact DLC --video V --out boxes.json [...]
  compare: python -B scripts/edge_parity.py compare boxes_246.json boxes_252.json
Gate (as the DETR preprocess A/B): >= 0.95 boxes matched in both directions (same class,
IoU >= 0.5) and median IoU >= 0.95.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def hottest_c():
    """Hottest readable thermal zone in C, or None (no zones / Windows). Some 252 zones raise EINVAL."""
    temps = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(zone.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps, default=None)


def cool_down(hot_c, cool_c):
    """Pause while the board is at hot_c or above, until it drops to cool_c (Edge252 has no heatsink)."""
    import time
    t = hottest_c()
    if t is None or t < hot_c:
        return 0.0
    began = time.monotonic()
    while t is not None and t > cool_c:
        time.sleep(5)
        t = hottest_c()
    return time.monotonic() - began


def dump(args):
    import cv2
    from benchmark.model import load_adapter
    detector, runtime = load_adapter(args.adapter, Path(args.artifact), imgsz=args.imgsz, layout=args.layout,
                                     confidence=args.confidence, allow_diagnostic=args.allow_diagnostic,
                                     nms_iou=args.nms_iou)
    capture = cv2.VideoCapture(args.video)
    frames, index, paused = [], 0, 0.0
    while len(frames) < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % args.stride == 0:
            hot_c = getattr(args, "hot_c", 0)  # optional for programmatic callers
            if hot_c and len(frames) % 50 == 0:
                paused += cool_down(hot_c, getattr(args, "cool_c", 65))
            if args.crop_bottom:  # live-transcoded video: Cloud drops the identity band before inference
                frame = frame[:-args.crop_bottom]
            boxes, _ms = detector.infer(frame)
            frames.append({"frame": index, "boxes": [[b.class_id, round(b.confidence, 5), b.left, b.top, b.right, b.bottom]
                                                     for b in boxes]})
        index += 1
    capture.release()
    if not frames:
        raise SystemExit("no frames decoded")
    Path(args.out).write_text(json.dumps({"artifact": args.artifact, "runtime": str(runtime.get("device")),
                                          "frames": frames}))
    print(f"DUMP frames={len(frames)} boxes={sum(len(f['boxes']) for f in frames)} "
          f"thermal_pause_s={paused:.0f} -> {args.out}")


def iou(a, b):
    w = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    h = max(0.0, min(a[5], b[5]) - max(a[3], b[3]))
    inter = w * h
    union = (a[4] - a[2]) * (a[5] - a[3]) + (b[4] - b[2]) * (b[5] - b[3]) - inter
    return inter / union if union > 0 else 0.0


def match(src, dst):
    """Greedy best same-class IoU >= 0.5 per src box; returns [(iou, score_diff)]."""
    used, out = set(), []
    for a in sorted(src, key=lambda x: -x[1]):
        best, j = 0.0, None
        for k, b in enumerate(dst):
            if k not in used and b[0] == a[0]:
                v = iou(a, b)
                if v > best:
                    best, j = v, k
        if j is not None and best >= .5:
            used.add(j)
            out.append((best, abs(a[1] - dst[j][1])))
    return out


def compare(a_path, b_path):
    a, b = (json.loads(Path(p).read_text())["frames"] for p in (a_path, b_path))
    pairs = {f["frame"]: f["boxes"] for f in b}
    common = [f for f in a if f["frame"] in pairs]
    n_a = sum(len(f["boxes"]) for f in common)
    n_b = sum(len(pairs[f["frame"]]) for f in common)
    ab = [m for f in common for m in match(f["boxes"], pairs[f["frame"]])]
    ba = [m for f in common for m in match(pairs[f["frame"]], f["boxes"])]
    identical = sum(f["boxes"] == pairs[f["frame"]] for f in common)
    keep_a = len(ab) / n_a if n_a else 1.0
    keep_b = len(ba) / n_b if n_b else 1.0
    ious = sorted(m[0] for m in ab)
    diffs = sorted(m[1] for m in ab)
    result = {"frames": len(common), "identical_frames": identical, "boxes_a": n_a, "boxes_b": n_b,
              "a_matched_in_b": round(keep_a, 4), "b_matched_in_a": round(keep_b, 4),
              "median_iou": round(statistics.median(ious), 4) if ious else None,
              "score_diff_p95": round(diffs[int(.95 * (len(diffs) - 1))], 4) if diffs else None}
    ok = bool(common) and keep_a >= .95 and keep_b >= .95 and (not ious or result["median_iou"] >= .95)
    print(json.dumps(result))
    print("PARITY_PASS" if ok else "PARITY_FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--adapter", required=True)
    d.add_argument("--artifact", required=True)
    d.add_argument("--video", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--frames", type=int, default=64)
    d.add_argument("--stride", type=int, default=20)
    d.add_argument("--imgsz", type=int, default=320)
    d.add_argument("--layout", default="nchw")
    d.add_argument("--confidence", type=float, default=.25)
    d.add_argument("--nms-iou", type=float, default=.5)
    d.add_argument("--allow-diagnostic", action="store_true")
    d.add_argument("--crop-bottom", type=int, default=0, help="rows to drop from the bottom of each frame")
    d.add_argument("--hot-c", type=float, default=0, help="pause at this temperature (0 = never)")
    d.add_argument("--cool-c", type=float, default=65, help="resume below this temperature")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    if args.cmd == "dump":
        dump(args)
        return 0
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
