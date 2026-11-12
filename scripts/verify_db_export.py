"""Edge -> DB correctness, from an export of the database (13/11, no web involved).

On 249 (one CSV per run, read only):
  docker exec v4-tsdb psql -U postgres -d v4 -c "\\copy (SELECT d.camera_id, d.stream_epoch, d.frame_id, d.media_pts_ns,
     f.model_mode, d.class_name, d.x1, d.y1, d.x2, d.y2 FROM detections d JOIN frames f USING (node_id, camera_id,
     stream_id, stream_epoch, frame_id) WHERE d.node_id='edge-252' AND d.capture_time > now() - interval '20 minutes')
     TO STDOUT CSV HEADER" > ~/db_export_252.csv
On the laptop:
  python -B scripts/verify_db_export.py db_export_252.csv --videos DIR --map cam-1=vn9 cam-2=vn6 cam-3=vn9

Per camera:
1. numbering: frame_id is the analytics frame the model saw; media_pts_ns the main stream frame the web would draw on.
   With frame-identical videos both name the same picture: (media_pts * fps) mod N == frame_id mod N (+ start frame).
   Reports how many frames agree and the offsets seen.
2. content: the stored boxes vs an independent detector (YOLO11m) on DIR/<video>.mp4 frame (frame_id mod N): share of
   stored vehicle / person boxes on one of its objects (IoU >= 0.5). Right boxes score high (models never agree 100 %),
   boxes of another moment score near 0.
"""
import argparse
import collections
import csv


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--map", nargs="+", required=True, help="cam-1=vn9 ...")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=15, help="frames per camera for the content check")
    ap.add_argument("--model", default="yolo11m.pt")
    a = ap.parse_args()
    video_of = dict(m.split("=") for m in a.map)
    keep = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    runs = collections.defaultdict(lambda: collections.defaultdict(list))   # (cam, epoch) -> frame_id -> rows
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        runs[(r["camera_id"], r["stream_epoch"])][int(r["frame_id"])].append(r)
    from ultralytics import YOLO
    net = YOLO(a.model)
    for (cam, epoch), frames in sorted(runs.items()):
        cap = cv2.VideoCapture(f"{a.videos}/{video_of[cam]}.mp4")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        offsets = collections.Counter()
        for fid, rows in frames.items():
            main_frame = round(int(rows[0]["media_pts_ns"]) * a.fps / 1e9)
            offsets[(main_frame - fid) % n] += 1
        agree = offsets.get(0, 0)
        mode = next(iter(frames.values()))[0]["model_mode"]
        print(f"{cam} epoch {epoch} {mode}: {len(frames)} frames in the DB; media_pts names the same picture as frame_id "
              f"in {agree}/{len(frames)}; most common offsets (frames): {offsets.most_common(3)}", flush=True)
        ids = sorted(frames)[:: max(1, len(frames) // a.frames)][: a.frames]
        hit = total = 0
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid % n)
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            ref = [b.xyxy[0].tolist() for r in net(img, verbose=False, conf=.25) for b in r.boxes
                   if r.names[int(b.cls)] in keep]
            for r in frames[fid]:
                if r["class_name"] not in keep:
                    continue
                box = (float(r["x1"]) * w, float(r["y1"]) * h, float(r["x2"]) * w, float(r["y2"]) * h)
                total += 1
                hit += any(iou(box, x) >= .5 for x in ref)
        share = hit / total if total else float("nan")
        print(f"    content: {total} stored boxes on {len(ids)} frames, on an object of frame_id's picture: {share:.1%} "
              f"-> {'OK' if share >= .5 else 'WRONG'}", flush=True)


if __name__ == "__main__":
    main()
