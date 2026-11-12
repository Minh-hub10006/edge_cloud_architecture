"""New vehicles per second per camera (plate OCR load: the Edge sends plate crops of every vehicle, the Cloud OCRs them).

  python -B scripts/new_vehicles.py DETRAC_TRAIN_XML_DIR DETRAC_TEST_XML_DIR  ->  results/new_vehicles.jsonl

DETRAC: track ids in the annotations (25 fps, 1 s windows like the profile table). VN-Hard: no ids in the auto labels
(15 fps), so ByteTrack (ultralytics, default bytetrack.yaml) counts tracks seen >= MIN_LIFE frames; the same tracker on
DETRAC boxes without their ids is checked against the true id count (printed, must agree within 20%).
"""
import glob
import json
import os
import re
import sys
import types
from argparse import Namespace

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_LIFE = 3


def _lap_shim():
    """ultralytics imports `lap` for its assignment; not installed here, so lapjv via scipy (no package install)."""
    def lapjv(cost, extend_cost=True, cost_limit=np.inf):
        x, y = np.full(cost.shape[0], -1), np.full(cost.shape[1], -1)
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] <= cost_limit:
                x[r], y[c] = c, r
        return 0.0, x, y
    sys.modules.setdefault("lap", types.SimpleNamespace(__version__="shim", lapjv=lapjv))


class _Dets:
    """The slice of ultralytics Results that BYTETracker.update reads."""

    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float)
        self.cls = np.zeros(len(self.conf))
        x1, y1, x2, y2 = self.xyxy.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.xyxy[idx], self.conf[idx])


def track_births(frames, fps):
    """frames: list of (frame index, [(x1, y1, x2, y2, conf)]) of the labelled frames -> first frame of each track
    seen in >= MIN_LIFE frames."""
    _lap_shim()
    from ultralytics.trackers.byte_tracker import BYTETracker
    tracker = BYTETracker(Namespace(tracker_type="bytetrack", track_high_thresh=.25, track_low_thresh=.1,
                                    new_track_thresh=.25, track_buffer=30, match_thresh=.8, fuse_score=True),
                          frame_rate=fps)
    first, seen = {}, {}
    for f, boxes in frames:
        b = np.asarray(boxes, float).reshape(-1, 5)
        for row in tracker.update(_Dets(b[:, :4], b[:, 4])):
            tid = int(row[4])
            first.setdefault(tid, f)
            seen[tid] = seen.get(tid, 0) + 1
    return [first[t] for t in first if seen[t] >= MIN_LIFE]


def detrac(xml_dir, split):
    rows, check = [], []
    for path in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        seq = os.path.basename(path).split(".")[0].replace("_v3", "")
        txt = open(path, encoding="utf-8", errors="ignore").read()
        first, frames = {}, []
        for m in re.finditer(r'<frame[^>]*num="(\d+)"(.*?)</frame>', txt, re.S):
            fr, boxes = int(m.group(1)) - 1, []
            for tid, body in re.findall(r'<target id="(\d+)"(.*?)</target>', m.group(2), re.S):
                first.setdefault(tid, fr)
                # the <box> tag only: <occlusion><region_overlap .../> carries left/top/width/height too
                b = dict(re.findall(r'(left|top|width|height)="([\d.]+)"', re.search(r"<box[^>]*>", body).group(0)))
                x, y, w, h = (float(b[k]) for k in ("left", "top", "width", "height"))
                boxes.append((x, y, x + w, y + h, 1.0))
            frames.append((fr, boxes))
        secs = frames[-1][0] // 25 + 1
        born = np.bincount([fr // 25 for fr in first.values()], minlength=secs)
        rows += [{"dataset": "detrac", "split": split, "sequence": seq, "second": s, "new_vehicles": int(c)}
                 for s, c in enumerate(born)]
        check.append((len(first), len(track_births(frames, 25))))
    return rows, check


def vn():
    rows = []
    for path in sorted(glob.glob("results/vn_hard/labels_auto/*.jsonl")):
        split, seq = os.path.basename(path)[:-6].split("_", 1)
        frames = [(r["frame"], [(*b[2:6], b[1]) for b in r["boxes"]])
                  for r in map(json.loads, open(path, encoding="utf-8"))]
        born = np.bincount([f // 15 for f in track_births(frames, 15)], minlength=frames[-1][0] // 15 + 1)
        rows += [{"dataset": "vn_hard", "split": split, "sequence": seq, "second": s, "new_vehicles": int(c)}
                 for s, c in enumerate(born)]
    return rows


def main():
    rows, check = [], []
    for xml_dir, split in ((sys.argv[1], "train"), (sys.argv[2], "test")):
        r, c = detrac(xml_dir, split)
        rows, check = rows + r, check + c
    true, tracked = map(np.array, zip(*check))
    ratio = tracked.sum() / true.sum()
    print(f"ByteTrack on DETRAC (ids hidden): {tracked.sum()} tracks vs {true.sum()} true ids, ratio {ratio:.2f}; "
          f"per-sequence ratio p10/p50/p90 {np.round(np.percentile(tracked / true, [10, 50, 90]), 2)}")
    assert .8 <= ratio <= 1.2, "tracker does not reproduce the true DETRAC vehicle count"
    rows += vn()
    with open("results/new_vehicles.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    for d, sp in (("detrac", "train"), ("detrac", "test"), ("vn_hard", "train"), ("vn_hard", "test")):
        v = [r["new_vehicles"] for r in rows if r["dataset"] == d and r["split"] == sp]
        print(f"{d} {sp}: {len(v)} s, new vehicles/s mean {np.mean(v):.2f} p90 {np.percentile(v, 90):.1f}")


if __name__ == "__main__":
    main()
