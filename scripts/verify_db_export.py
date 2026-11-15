"""Edge -> DB content check of one run, every table, as a web reader would use it (15/11; no web involved).

On 249:   bash scripts/export_run.sh edge-252 15            -> ~/run_export_edge-252.tgz (unpack on the laptop)
Laptop:   python -B scripts/verify_db_export.py run_export_edge-252 --videos DIR --main-videos DIR2 \
              --map cam-1=vn6:N cam-2=vn9:A cam-3=vn6:B cam-4=vn9:D [--model yolo11m.pt]
          DIR: the analytics videos the Edge played (<video>.mp4); DIR2: the main streams (<video>_main.mp4, with the
          frame-number band). The newest run (stream epoch) of each camera is checked.

Checks (gate in brackets), per camera:
  frames / detections
    numbering   media_pts_ns names the same picture as frame_id                                   [100 %]
    band        the main stream's frame at media_pts carries that frame number in its identity band, and its
                picture is the analytics picture of frame_id (what the web will draw on)          [20/20 sampled]
    coords      0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1                                           [100 %]
    frames      one frames row per frame; every detection's frame is there; box_count = its boxes [100 %]
    order       capture time increases with frame_id                                               [>= 99.9 %]
    mode        model_mode is the camera's mode                                                    [100 %]
    objects     stored vehicle / person boxes on an object of an independent detector (YOLO11m)   [>= 50 %]
    classes     ... of those, the same class group (person-bicycle-motorcycle, car-bus-truck)       [>= 85 %]
    tracks      a track's box in its next frame (<= 3 frames later) overlaps it (IoU >= 0.3)       [>= 95 %]
  counts      a window closes at the first result past its length; per-class means rebuilt from detections in the same window    [>= 90 % within
              max(0.5, 10 %)]
  events      known type; end >= start; snapshot present and a picture of that camera's video      [100 %]
  plates      crop inside its vehicle's box on its frame [>= 90 %]; valid flag = plate format [100 %]; its track
              is in detections [>= 99 %]
  crowd       frame stored [100 %]; persons_boxes = person boxes of that frame [>= 95 %]; grid sums to the
              estimate within 1 [>= 95 %]
  actions     frame stored [>= 95 %]; score in 0..1 [100 %]
Writes <dir>/verify_result.json (every check with value, gate and verdict) and prints ALL PASS / SOME FAIL.
"""
import argparse
import bisect
import collections
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODES = {"N": "yolo_edge", "A": "yolo11s_edge", "B": "rtdetr_split_cloud", "C": "rtdetr_cloud", "D": "dfine_x_cloud"}
KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
GROUP = {"person": 0, "bicycle": 0, "motorcycle": 0, "car": 1, "bus": 1, "truck": 1}
COUNT_COLS = {"persons": "person", "bicycles": "bicycle", "cars": "car", "motorcycles": "motorcycle", "buses": "bus",
              "trucks": "truck"}
EVENT_TYPES = {"congestion", "crowd", "gathering", "fight", "crash", "loiter"}


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def rows(d, name):
    p = os.path.join(d, name + ".csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


class Report:
    def __init__(self):
        self.items = []

    def add(self, cam, check, value, gate, ok, detail=""):
        self.items.append({"camera": cam, "check": check, "value": value, "gate": gate, "pass": bool(ok),
                           "detail": detail})
        print(f"  {'ok  ' if ok else 'FAIL'} {cam:6s} {check:10s} {value:>10s}  (gate {gate}) {detail}", flush=True)

    def share(self, cam, check, good, total, need, detail="", rows=0):
        """rows: table rows behind an image check - rows but no image examined fails (15/11: 0/0 passed silently)."""
        v = good / total if total else 1.0
        if rows and not total:
            detail = f"{rows} rows but no image to check (export copied none of this run?) {detail}"
        self.add(cam, check, f"{good}/{total}", f">= {need:.1%}" if need < 1 else "100 %",
                 v >= need and not (rows and not total), detail)


class Videos:
    def __init__(self, analytics_dir, main_dir):
        import cv2
        self.cv2, self.adir, self.mdir, self.caps, self.thumbs = cv2, analytics_dir, main_dir, {}, {}

    def _cap(self, path):
        if path not in self.caps:
            self.caps[path] = self.cv2.VideoCapture(path)
        return self.caps[path]

    def count(self, video, main=False):
        return int(self._cap(self._path(video, main)).get(self.cv2.CAP_PROP_FRAME_COUNT))

    def _path(self, video, main):
        return os.path.join(self.mdir, video + "_main.mp4") if main else os.path.join(self.adir, video + ".mp4")

    def frame(self, video, index, main=False):
        cap = self._cap(self._path(video, main))
        cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
        return img if ok else None

    def thumbnails(self, video):
        """Every analytics frame shrunk to 72x128 grey (for matching event snapshots)."""
        if video not in self.thumbs:
            cap, out = self.cv2.VideoCapture(self._path(video, False)), []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                out.append(self.cv2.resize(self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY), (72, 128),
                                           interpolation=self.cv2.INTER_AREA).astype("float32"))
            self.thumbs[video] = out
        return self.thumbs[video]


def main():
    import cv2
    import numpy as np
    from media.copy_publisher import MAIN_KEY
    from media.frame_identity import read_identity
    from inference.plate_ocr import valid_plate
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--main-videos", required=True)
    ap.add_argument("--map", nargs="+", required=True, help="cam-1=vn9:N ...")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=30, help="frames per camera for the object check")
    ap.add_argument("--model", default="yolo11m.pt")
    a = ap.parse_args()
    spec = {m.split("=")[0]: m.split("=")[1].split(":") for m in a.map}   # cam -> [video, mode letter]
    vids = Videos(a.videos, a.main_videos)
    rep = Report()

    # ------------------------------------------------------------------ newest run of each camera
    det_all, frm_all = rows(a.dir, "detections"), rows(a.dir, "frames")
    epoch = {}
    for r in frm_all + det_all:
        epoch[r["camera_id"]] = max(epoch.get(r["camera_id"], 0), int(r["stream_epoch"]))
    keep = lambda r: int(r["stream_epoch"]) == epoch.get(r["camera_id"])   # noqa: E731
    det = collections.defaultdict(lambda: collections.defaultdict(list))   # cam -> frame_id -> boxes
    for r in filter(keep, det_all):
        det[r["camera_id"]][int(r["frame_id"])].append(r)
    frm = collections.defaultdict(list)
    for r in filter(keep, frm_all):
        frm[r["camera_id"]].append(r)
    from ultralytics import YOLO
    net = YOLO(a.model)

    for cam in sorted(set(spec) | set(frm)):
        if cam not in spec:
            rep.add(cam, "map", "-", "camera in --map", False, "camera stored but not in --map")
            continue
        video, letter = spec[cam]
        n, n_main = vids.count(video), vids.count(video, main=True)
        frames, boxes = frm.get(cam, []), det.get(cam, {})
        print(f"== {cam} ({video}, {letter} = {MODES[letter]}): {len(frames)} frames rows, "
              f"{sum(map(len, boxes.values()))} boxes, epoch {epoch.get(cam)}", flush=True)
        if not frames:
            rep.add(cam, "frames", "0", "> 0", False, "no frames stored for this camera")
            continue
        fids = sorted(int(r["frame_id"]) for r in frames)
        # numbering
        good = sum((round(int(r["media_pts_ns"]) * a.fps / 1e9) - int(r["frame_id"])) % n == 0 for r in frames)
        rep.share(cam, "numbering", good, len(frames), 1.0)
        # band: what the web reads from the main stream
        by_fid = {int(r["frame_id"]): r for r in frames}
        sample = fids[:: max(1, len(fids) // 20)][:20]
        good, bad = 0, []
        for fid in sample:
            pos = round(int(by_fid[fid]["media_pts_ns"]) * a.fps / 1e9) % n_main
            img = vids.frame(video, pos, main=True)
            try:
                seq = read_identity(img, key=MAIN_KEY).seq
            except (ValueError, TypeError) as exc:
                bad.append(f"{fid}: {exc}")
                continue
            ana = vids.frame(video, fid % n)
            pic = cv2.resize(img[:-64], (ana.shape[1], ana.shape[0]), interpolation=cv2.INTER_AREA)
            diff = float(np.abs(pic.astype(int) - ana.astype(int)).mean())
            if seq == pos and seq == fid % n and diff < 12:
                good += 1
            else:
                bad.append(f"{fid}: band {seq} pos {pos} diff {diff:.1f}")
        rep.share(cam, "band", good, len(sample), 1.0, "; ".join(bad[:3]))
        # coords
        allb = [b for fb in boxes.values() for b in fb]
        okc = sum(0 <= float(b["x1"]) < float(b["x2"]) <= 1.001 and 0 <= float(b["y1"]) < float(b["y2"]) <= 1.001
                  for b in allb)
        rep.share(cam, "coords", okc, len(allb), 1.0)
        # frames table consistency
        dup = len(fids) - len(set(fids))
        orphan = sum(1 for f in boxes if f not in by_fid)
        miscount = sum(1 for f, r in by_fid.items() if int(r["box_count"]) != len(boxes.get(f, [])))
        rep.add(cam, "frames", f"{dup}/{orphan}/{miscount}", "0 dup / 0 orphan / 0 count", dup == orphan == miscount == 0,
                "duplicate frames / boxes without a frame row / box_count != boxes")
        # order
        caps = [float(by_fid[f]["cap_ms"]) for f in sorted(by_fid)]
        rep.share(cam, "order", sum(b >= a_ for a_, b in zip(caps, caps[1:])), max(1, len(caps) - 1), .999)
        # mode
        rep.share(cam, "mode", sum(r["model_mode"] == MODES[letter] for r in frames), len(frames), 1.0)
        # objects / classes (independent detector on the analytics picture)
        hit = total = same = 0
        for fid in fids[:: max(1, len(fids) // a.frames)][: a.frames]:
            img = vids.frame(video, fid % n)
            h, w = img.shape[:2]
            ref = [(b.xyxy[0].tolist(), r.names[int(b.cls)]) for r in net(img, verbose=False, conf=.25) for b in r.boxes
                   if r.names[int(b.cls)] in KEEP]
            for b in boxes.get(fid, []):
                if b["class_name"] not in KEEP:
                    continue
                total += 1
                box = (float(b["x1"]) * w, float(b["y1"]) * h, float(b["x2"]) * w, float(b["y2"]) * h)
                best = max(ref, key=lambda x: iou(box, x[0]), default=None)
                if best is not None and iou(box, best[0]) >= .5:
                    hit += 1
                    same += GROUP[best[1]] == GROUP[b["class_name"]]
        rep.share(cam, "objects", hit, total, .5)
        rep.share(cam, "classes", same, hit, .85)
        # tracks: the same id a few frames later still overlaps
        last, pairs, good = {}, 0, 0
        for fid in sorted(boxes):
            for b in boxes[fid]:
                if not b["track_id"]:
                    continue
                box = tuple(float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                prev = last.get(b["track_id"])
                if prev is not None and fid - prev[0] <= 3:
                    pairs += 1
                    good += iou(prev[1], box) >= .3
                last[b["track_id"]] = (fid, box)
        rep.share(cam, "tracks", good, pairs, .95)

    # ------------------------------------------------------------------ counts
    cnt = collections.defaultdict(list)
    for r in rows(a.dir, "counts"):
        if keep(r):
            cnt[r["camera_id"]].append(r)
    for cam, ws in sorted(cnt.items()):
        ws.sort(key=lambda r: float(r["end_ms"]))
        ends = [float(r["end_ms"]) for r in ws]
        caps = sorted((float(r["cap_ms"]), int(r["frame_id"])) for r in frm.get(cam, []))
        # 15/11: a window closes at the first result past its length; with no result for seconds (overload) it is
        # longer and that is right. Late = a stored frame came after the window's length but it still did not close.
        times, late, longest = [t for t, _ in caps], 0, 0.0
        for a_, b in zip(ends, ends[1:]):
            longest = max(longest, b - a_)
            limit = a_ + float(ws[0]["window_s"]) * 1000 + 500
            late += any(limit < t < b - 1 for t in times[bisect.bisect_right(times, limit):][:1])
        rep.share(cam, "count gap", max(1, len(ends) - 1) - late, max(1, len(ends) - 1), 1.0,
                  f"longest window {longest / 1000:.1f} s")
        good = checked = 0
        worst = ""
        for prev, r in zip(ws, ws[1:]):
            lo, hi = float(prev["end_ms"]), float(r["end_ms"])
            inside = [f for t, f in caps if lo < t <= hi + 0.5]
            if not inside:
                continue
            checked += 1
            errs = []
            for col, name in COUNT_COLS.items():
                mean = sum(sum(b["class_name"] == name for b in det[cam].get(f, [])) for f in inside) / len(inside)
                errs.append((abs(mean - float(r[col])), max(.5, .1 * float(r[col])), col, mean, float(r[col])))
            ok = all(e <= tol for e, tol, *_ in errs)
            good += ok
            if not ok and not worst:
                e = max(errs, key=lambda x: x[0] - x[1])
                worst = f"window ending {r['end_ms'][:13]}: {e[2]} DB {e[3]:.2f} vs counts {e[4]:.2f}"
        rep.share(cam, "counts", good, checked, .9, worst)

    # ------------------------------------------------------------------ events
    evs = [r for r in rows(a.dir, "events") if keep(r)]
    for cam in sorted({r["camera_id"] for r in evs}):
        mine = [r for r in evs if r["camera_id"] == cam]
        rep.share(cam, "event type", sum(r["event_type"] in EVENT_TYPES for r in mine), len(mine), 1.0)
        rep.share(cam, "event time", sum(not r["ended_at"] or r["ended_at"] >= r["started_at"] for r in mine),
                  len(mine), 1.0)
        good, total, bad = 0, 0, []
        for r in mine:
            f = os.path.join(a.dir, "media", os.path.basename(r["snapshot_path"] or ""))
            if not r["snapshot_path"] or not os.path.exists(f):
                continue
            total += 1
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            t = cv2.resize(img, (72, 128), interpolation=cv2.INTER_AREA).astype("float32")
            best = min(float(np.abs(t - x).mean()) for x in vids.thumbnails(spec[cam][0]))
            good += best < 12
            if best >= 12:
                bad.append(f"{os.path.basename(f)} best diff {best:.1f}")
        rep.share(cam, "snapshot", good, total, 1.0, "; ".join(bad[:2]), rows=sum(bool(r["snapshot_path"]) for r in mine))

    # ------------------------------------------------------------------ plates
    pls = [r for r in rows(a.dir, "plates") if keep(r)]
    for cam in sorted({r["camera_id"] for r in pls}):
        mine = [r for r in pls if r["camera_id"] == cam]
        found = checked = 0
        for p in mine:
            f = os.path.join(a.dir, "media", os.path.basename(p["image_path"] or ""))
            if not p["image_path"] or not os.path.exists(f):
                continue
            crop, img = cv2.imread(f), vids.frame(spec[cam][0], int(p["frame_id"]) % vids.count(spec[cam][0]))
            veh = [b for b in det[cam].get(int(p["frame_id"]), []) if b["track_id"] == p["track_id"]]
            checked += 1
            if not veh or crop is None or crop.shape[0] > img.shape[0] or crop.shape[1] > img.shape[1]:
                continue
            _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(img, crop, cv2.TM_CCOEFF_NORMED))
            h, w = img.shape[:2]
            v = veh[0]
            inside = (float(v["x1"]) * w - 8 <= x and x + crop.shape[1] <= float(v["x2"]) * w + 8
                      and float(v["y1"]) * h - 8 <= y and y + crop.shape[0] <= float(v["y2"]) * h + 8)
            found += score >= .8 and inside
        rep.share(cam, "plate crop", found, checked, .9, rows=sum(bool(p["image_path"]) for p in mine))
        valid = lambda s: s in ("t", "true", "True")   # noqa: E731
        rep.share(cam, "plate flag", sum(valid(p["valid"]) == valid_plate(p["text"]) for p in mine), len(mine), 1.0)
        tracks = {b["track_id"] for fb in det[cam].values() for b in fb}
        rep.share(cam, "plate track", sum(p["track_id"] in tracks for p in mine), len(mine), .99)

    # ------------------------------------------------------------------ crowd
    crs = [r for r in rows(a.dir, "crowd") if keep(r)]
    for cam in sorted({r["camera_id"] for r in crs}):
        mine = [r for r in crs if r["camera_id"] == cam]
        stored = {int(r["frame_id"]) for r in frm.get(cam, [])}
        rep.share(cam, "crowd frame", sum(int(r["frame_id"]) in stored for r in mine), len(mine), 1.0)
        rep.share(cam, "crowd boxes", sum(r["persons_boxes"] != "" and int(r["persons_boxes"]) == sum(
            b["class_name"] == "person" for b in det[cam].get(int(r["frame_id"]), [])) for r in mine), len(mine), .95)
        rep.share(cam, "crowd grid", sum(abs(sum(map(sum, json.loads(r["grid"]))) - float(r["persons_est"])) <= 1
                                         for r in mine), len(mine), .95)

    # ------------------------------------------------------------------ actions
    acs = [r for r in rows(a.dir, "actions") if keep(r)]
    for cam in sorted({r["camera_id"] for r in acs}):
        mine = [r for r in acs if r["camera_id"] == cam]
        stored = {int(r["frame_id"]) for r in frm.get(cam, [])}
        rep.share(cam, "action frame", sum(int(r["frame_id"]) in stored for r in mine), len(mine), .95)
        rep.share(cam, "action score", sum(0 <= float(r["score"]) <= 1 for r in mine), len(mine), 1.0)

    ok = all(i["pass"] for i in rep.items)
    Path(a.dir, "verify_result.json").write_text(json.dumps({"pass": ok, "checks": rep.items}, indent=1),
                                                 encoding="utf-8")
    print("== ALL PASS" if ok else "== SOME FAIL", f"({sum(not i['pass'] for i in rep.items)} failed of {len(rep.items)})")


if __name__ == "__main__":
    main()
