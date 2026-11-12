"""Edge -> DB correctness check (13/11): are the stored boxes on the right objects of the right frame?

  python -B scripts/verify_db_boxes.py NODE --videos DIR [--url http://192.168.180.249:18200] [--frames 12]

For each stream of NODE now running (check page API, read only): takes the stored boxes of several frames, cuts the
same frame (media_pts -> frame number) out of DIR/<video>_main.mp4, runs an independent detector (YOLO11m, Ultralytics)
on it and reports the share of stored vehicle/person boxes that overlap one of its boxes (IoU >= 0.5). Boxes put on the
wrong frame score near 0 (12/11: the analytics files were a 24 s cut of the 140 s main files); right ones score high
(different models never agree 100 %). Which video each camera plays is given as CAM=VIDEO (default from the run spec
"vn9:N vn6:B vn9:D": cam-1=vn9 cam-2=vn6 cam-3=vn9).
"""
import argparse
import json
import urllib.parse
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def main():
    import cv2
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("node")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--url", default="http://192.168.180.249:18200")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--map", nargs="*", default=["cam-1=vn9", "cam-2=vn6", "cam-3=vn9", "cam-4=vn6", "cam-5=vn9"])
    ap.add_argument("--model", default="yolo11m.pt")
    a = ap.parse_args()
    video_of = dict(m.split("=") for m in a.map)
    keep = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    net = YOLO(a.model)
    for s in [s for s in get(a.url + "/api/streams") if s["node_id"] == a.node]:
        q = urllib.parse.urlencode({"node": a.node, "camera": s["camera_id"], "stream": s["stream_id"],
                                    "epoch": s["stream_epoch"], "after_cap_ms": -1})
        frames = [f for f in get(a.url + "/api/recent?" + q)["frames"] if f["boxes"]]
        frames = frames[:: max(1, len(frames) // a.frames)][: a.frames]
        cap = cv2.VideoCapture(f"{a.videos}/{video_of[s['camera_id']]}_main.mp4")
        hit = total = 0
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(f["pts_ms"] * 15 / 1000))
            ok, img = cap.read()
            if not ok:
                continue
            img = img[:-64]   # identity band
            h, w = img.shape[:2]
            ref = [b.xyxy[0].tolist() for r in net(img, verbose=False, conf=.25) for b in r.boxes
                   if r.names[int(b.cls)] in keep]
            for b in f["boxes"]:
                if b["class_name"] not in keep:
                    continue
                box = (b["x1"] * w, b["y1"] * h, b["x2"] * w, b["y2"] * h)
                total += 1
                hit += any(iou(box, r) >= .5 for r in ref)
        share = hit / total if total else float("nan")
        print(f"{a.node} {s['camera_id']} {s['mode']:<20s} frames {len(frames):2d}  stored boxes {total:4d}  "
              f"on an object {share:6.1%}  -> {'OK' if share >= .5 else 'WRONG FRAME / POSITION'}", flush=True)


if __name__ == "__main__":
    main()
