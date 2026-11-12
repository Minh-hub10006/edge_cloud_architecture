"""Edge -> DB content check of one run (13/11, no web involved).

On 249:     bash scripts/export_run.sh edge-252 10          -> ~/run_export_edge-252.tgz (unpack on the laptop)
Laptop:     python -B scripts/verify_db_export.py run_export_edge-252 --videos DIR --map cam-1=vn9 cam-2=vn6 cam-3=vn9
            (DIR holds the analytics videos <video>.mp4 the Edge played: same pictures, frame for frame)

Pass gates, per camera:
1. fast lane: frame_id and media_pts_ns name the same picture on every frame; stored vehicle / person boxes vs an
   independent detector (YOLO11m) on that picture: >= 50 % on one of its objects (IoU >= 0.5). Models never agree
   100 %; boxes of another picture or a shifted one score < 20 % (252 before the 13/11 decoder fix: 2-6 %).
2. UC1: each plate crop is found in its frame (frame_id) by template matching (score >= 0.8) inside the stored box of
   its vehicle (track_id): >= 90 % of the crops.
3. the other tables: rows per camera (counts every 5 s, events / crowd / actions when the scene triggers them), and a
   contact sheet of event snapshots + plate crops (<dir>/sheet.jpg) to look at.
"""
import argparse
import collections
import csv
import os


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def rows(d, name):
    p = os.path.join(d, name + ".csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    import cv2
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--map", nargs="+", required=True, help="cam-1=vn9 ...")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=30, help="frames per camera for the box check")
    ap.add_argument("--model", default="yolo11m.pt")
    a = ap.parse_args()
    video_of = dict(m.split("=") for m in a.map)
    keep = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    caps = {}

    def picture(cam, fid):
        if cam not in caps:
            caps[cam] = cv2.VideoCapture(f"{a.videos}/{video_of[cam]}.mp4")
        c = caps[cam]
        c.set(cv2.CAP_PROP_POS_FRAMES, fid % int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
        return c.read()[1]

    runs = collections.defaultdict(lambda: collections.defaultdict(list))   # (cam, epoch) -> frame_id -> boxes
    for r in rows(a.dir, "detections"):
        runs[(r["camera_id"], r["stream_epoch"])][int(r["frame_id"])].append(r)
    from ultralytics import YOLO
    net = YOLO(a.model)
    verdict = []
    print("== fast lane (detections)")
    for (cam, epoch), frames in sorted(runs.items()):
        n = int(cv2.VideoCapture(f"{a.videos}/{video_of[cam]}.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        agree = sum((round(int(b[0]["media_pts_ns"]) * a.fps / 1e9) - fid) % n == 0 for fid, b in frames.items())
        hit = total = 0
        for fid in sorted(frames)[:: max(1, len(frames) // a.frames)][: a.frames]:
            img = picture(cam, fid)
            h, w = img.shape[:2]
            ref = [b.xyxy[0].tolist() for r in net(img, verbose=False, conf=.25) for b in r.boxes
                   if r.names[int(b.cls)] in keep]
            for r in frames[fid]:
                if r["class_name"] in keep:
                    total += 1
                    box = (float(r["x1"]) * w, float(r["y1"]) * h, float(r["x2"]) * w, float(r["y2"]) * h)
                    hit += any(iou(box, x) >= .5 for x in ref)
        share = hit / total if total else 0.0
        ok = agree == len(frames) and share >= .5
        verdict.append(ok)
        print(f"{cam} ({video_of[cam]}, {next(iter(frames.values()))[0]['model_mode']}): {len(frames)} frames, numbering "
              f"{agree}/{len(frames)}, boxes on an object {share:.0%} of {total} -> {'OK' if ok else 'FAIL'}", flush=True)

    plates = rows(a.dir, "plates")
    print(f"== UC1 plates: {len(plates)} reads")
    if plates:
        found = checked = 0
        for p in plates:
            f = os.path.join(a.dir, "media", os.path.basename(p["image_path"] or ""))
            if not p["image_path"] or not os.path.exists(f):
                continue
            crop, img = cv2.imread(f), picture(p["camera_id"], int(p["frame_id"]))
            veh = [r for r in runs[(p["camera_id"], p["stream_epoch"])].get(int(p["frame_id"]), [])
                   if r["track_id"] == p["track_id"]]
            checked += 1
            if not veh or crop is None or crop.shape[0] > img.shape[0] or crop.shape[1] > img.shape[1]:
                continue
            res = cv2.matchTemplate(img, crop, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            h, w = img.shape[:2]
            v = veh[0]
            inside = (float(v["x1"]) * w - 8 <= x and x + crop.shape[1] <= float(v["x2"]) * w + 8
                      and float(v["y1"]) * h - 8 <= y and y + crop.shape[0] <= float(v["y2"]) * h + 8)
            found += score >= .8 and inside
        share = found / checked if checked else 0.0
        verdict.append(share >= .9)
        print(f"crops found in their frame, inside their vehicle's box: {found}/{checked} ({share:.0%}) -> "
              f"{'OK' if share >= .9 else 'FAIL'}")
        texts = collections.Counter(p["text"] for p in plates if p["valid"] in ("t", "true", "True"))
        print(f"valid texts (OCR accuracy is the CV team's): {sum(texts.values())}, e.g. {list(texts)[:6]}")

    print("== other tables (rows per camera)")
    for name in ("counts", "events", "crowd", "actions"):
        t = rows(a.dir, name)
        by = collections.Counter(r["camera_id"] for r in t)
        kinds = collections.Counter(r.get("event_type") or r.get("check_type") or "" for r in t)
        print(f"{name:8s} {len(t):5d}  {dict(sorted(by.items()))}  {dict(kinds) if kinds.get('') != len(t) else ''}")
    for r in rows(a.dir, "crowd")[:5]:
        print(f"   crowd {r['camera_id']} frame {r['frame_id']}: boxes {r['persons_boxes']} estimate "
              f"{float(r['persons_est']):.1f}")

    pics = [cv2.imread(os.path.join(a.dir, "media", f)) for f in sorted(os.listdir(os.path.join(a.dir, "media")))]
    pics = [p for p in pics if p is not None]
    if pics:
        tiles = [cv2.resize(p, (160, max(40, round(p.shape[0] * 160 / p.shape[1]))))[:240] for p in pics[:48]]
        cols = [np.vstack(tiles[i::8]) for i in range(min(8, len(tiles)))]
        hh = max(c.shape[0] for c in cols)
        sheet = np.hstack([np.pad(c, ((0, hh - c.shape[0]), (0, 0), (0, 0))) for c in cols])
        cv2.imwrite(os.path.join(a.dir, "sheet.jpg"), sheet)
        print(f"contact sheet: {os.path.join(a.dir, 'sheet.jpg')} ({len(tiles)} images)")
    print("== ALL PASS" if all(verdict) else "== SOME FAIL")


if __name__ == "__main__":
    main()
