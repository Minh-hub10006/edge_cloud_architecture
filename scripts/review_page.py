"""A static review page of one run, per use case, to check by eye (13/11). Every stored result is drawn on its own
frame of the test video: no live video, nothing to synchronise.

  python -B scripts/review_page.py RUN_EXPORT_DIR --videos DIR --map cam-1=vn9 cam-2=vn6 ... [--out DIR/review]
  -> <out>/index.html (open in a browser; images next to it)

RUN_EXPORT_DIR: unpacked scripts/export_run.sh output. DIR: the analytics videos the Edge played (<video>.mp4).
"""
import argparse
import collections
import csv
import html
import json
import os

COLORS = {"person": (0, 200, 255), "car": (0, 255, 0), "motorcycle": (255, 160, 0), "bus": (255, 0, 255),
          "truck": (0, 0, 255), "bicycle": (255, 255, 0)}


def rows(d, name):
    p = os.path.join(d, name + ".csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    import cv2
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--map", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--per-camera", type=int, default=6, help="fast-lane frames per camera")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "review")
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    video_of = dict(m.split("=") for m in a.map)
    caps = {}

    def frame(cam, fid):
        if cam not in caps:
            caps[cam] = cv2.VideoCapture(os.path.join(a.videos, video_of[cam] + ".mp4"))
        c = caps[cam]
        c.set(cv2.CAP_PROP_POS_FRAMES, fid % int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
        return c.read()[1]

    det = collections.defaultdict(lambda: collections.defaultdict(list))   # (cam, epoch) -> frame_id -> boxes
    for r in rows(a.dir, "detections"):
        det[(r["camera_id"], r["stream_epoch"])][int(r["frame_id"])].append(r)
    mode_of = {k: next(iter(v.values()))[0]["model_mode"] for k, v in det.items()}

    def draw(img, boxes, only=None, label_track=False):
        h, w = img.shape[:2]
        for b in boxes:
            if only and b["class_name"] not in only:
                continue
            p1 = (int(float(b["x1"]) * w), int(float(b["y1"]) * h))
            p2 = (int(float(b["x2"]) * w), int(float(b["y2"]) * h))
            cv2.rectangle(img, p1, p2, COLORS.get(b["class_name"], (200, 200, 200)), 3)
            if label_track and b["track_id"]:
                cv2.putText(img, b["track_id"], (p1[0], p1[1] - 6), 0, 1.1, (255, 255, 255), 3)
        return img

    n_img = [0]

    def save(img, width=360):
        n_img[0] += 1
        name = f"img/{n_img[0]:05d}.jpg"
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return name

    def card(src, text, cls=""):
        return (f'<figure class="{cls}"><a href="{src}" target="_blank"><img src="{src}" loading="lazy"></a>'
                f"<figcaption>{text}</figcaption></figure>")

    sec = []
    # fast lane
    body = []
    for (cam, epoch), frames in sorted(det.items()):
        ids = sorted(frames)
        picks = ids[len(ids) // (a.per_camera + 1):: max(1, len(ids) // (a.per_camera + 1))][: a.per_camera]
        cards = [card(save(draw(frame(cam, f), frames[f], label_track=True)),
                      f"khung {f % 2100} · {len(frames[f])} box") for f in picks]
        body.append(f"<h3>{cam} · {video_of[cam]} · {mode_of[(cam, epoch)]} · {len(frames)} khung trong DB</h3>"
                    f'<div class="grid">{"".join(cards)}</div>')
    sec.append(("fast", "Làn nhanh: box trong DB vẽ lên đúng khung",
                "Mỗi box phải ôm đúng một xe / người. Số trên box là mã bám (cùng xe giữ cùng số qua các khung).", body))

    # UC1
    body, plates = [], rows(a.dir, "plates")
    for p in plates:
        f = os.path.join(a.dir, "media", os.path.basename(p["image_path"] or ""))
        if not p["image_path"] or not os.path.exists(f):
            continue
        img = frame(p["camera_id"], int(p["frame_id"]))
        veh = [b for b in det[(p["camera_id"], p["stream_epoch"])].get(int(p["frame_id"]), [])
               if b["track_id"] == p["track_id"]]
        h, w = img.shape[:2]
        if veh:
            v = veh[0]
            x1, y1 = int(float(v["x1"]) * w), int(float(v["y1"]) * h)
            x2, y2 = int(float(v["x2"]) * w), int(float(v["y2"]) * h)
            vimg = img[max(0, y1):y2, max(0, x1):x2].copy()
        else:
            vimg = img.copy()
        crop = cv2.imread(f)
        ok = "✓ hợp lệ" if p["valid"] in ("t", "true", "True") else "✗ không hợp lệ"
        body.append(f'<div class="pair">{card(save(vimg, 220), f"xe #{p["track_id"]} · {p["camera_id"]} · khung {int(p["frame_id"]) % 2100}")}'
                    f'{card(save(crop, 220), f"<b>{html.escape(p["text"])}</b> · {ok} · lần đọc {p["read_no"]}")}</div>')
    sec.append(("uc1", f"UC1 biển số: {len(plates)} lần đọc ({len(body)} có ảnh)",
                "Trái: xe (cắt theo box của xe trong DB, đúng khung). Phải: ảnh biển Edge gửi lên và chữ Cloud đọc. "
                "Ảnh biển phải là biển của chính xe bên trái. Đọc đúng chữ hay không là việc của nhóm CV.",
                [f'<div class="grid wide">{"".join(body)}</div>']))

    # UC2 counts + congestion
    counts = rows(a.dir, "counts")
    by = collections.defaultdict(list)
    for c in counts:
        by[c["camera_id"]].append(c)
    trs = []
    for cam, cs in sorted(by.items()):
        cs.sort(key=lambda c: c["window_end"])
        avg = lambda k: sum(float(c[k]) for c in cs) / len(cs)
        st = [float(c["stationary_share"]) for c in cs if c["stationary_share"]]
        trs.append(f"<tr><td>{cam}</td><td>{len(cs)}</td><td>{avg('cars'):.1f}</td><td>{avg('motorcycles'):.1f}</td>"
                   f"<td>{avg('buses'):.1f}</td><td>{avg('trucks'):.1f}</td><td>{avg('persons'):.1f}</td>"
                   f"<td>{max(int(c['vehicles_max']) for c in cs)}</td><td>{sum(int(c['new_tracks']) for c in cs)}</td>"
                   f"<td>{(sum(st) / len(st) if st else 0):.0%}</td></tr>")
    table = ("<table><tr><th>camera</th><th>cửa sổ 5 s</th><th>ô tô TB</th><th>xe máy TB</th><th>buýt TB</th>"
             "<th>tải TB</th><th>người TB</th><th>xe tối đa</th><th>xe mới</th><th>đứng yên</th></tr>"
             + "".join(trs) + "</table>")
    sec.append(("uc2", "UC2 đếm và ùn tắc", "Số trung bình mỗi khung trong từng cửa sổ 5 s. So với ảnh ở phần làn "
                "nhanh của cùng camera (đếm box bằng mắt). Sự kiện ùn tắc (nếu có) ở phần Sự kiện.", [table]))

    # UC3 crowd
    body = []
    for c in rows(a.dir, "crowd")[:24]:
        img = frame(c["camera_id"], int(c["frame_id"]))
        h, w = img.shape[:2]
        grid = np.array(json.loads(c["grid"]), np.float32)
        heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_NEAREST)
        over = img.copy()
        over[heat >= 1] = (0.55 * over[heat >= 1] + 0.45 * np.array((0, 0, 255))).astype(np.uint8)
        for i in range(8):
            for j in range(8):
                if grid[i, j] >= .5:
                    cv2.putText(over, f"{grid[i, j]:.0f}", (int((j + .3) * w / 8), int((i + .6) * h / 8)), 0, 1.6,
                                (255, 255, 255), 4)
        boxes = det[(c["camera_id"], c["stream_epoch"])].get(int(c["frame_id"]), [])
        draw(over, boxes, only={"person"})
        body.append(card(save(over), f"{c['camera_id']} · khung {int(c['frame_id']) % 2100} · DM-Count "
                         f"<b>{float(c['persons_est']):.0f}</b> người · box người {c['persons_boxes']}"))
    sec.append(("uc3", "UC3 đám đông (DM-Count)", "Đỏ: ô có người theo DM-Count (số = người ước lượng trong ô 1/8 x "
                "1/8). Khung vàng: box người của làn nhanh. DM-Count đếm cả người ngồi trên xe máy, box người thì không "
                "nên hai số khác nhau là bình thường; vùng đỏ phải nằm đúng chỗ có người.",
                [f'<div class="grid">{"".join(body)}</div>']))

    # UC4
    body = []
    for c in sorted(rows(a.dir, "actions"), key=lambda r: -float(r["score"]))[:24]:
        img = frame(c["camera_id"], int(c["frame_id"]))
        boxes = det[(c["camera_id"], c["stream_epoch"])].get(int(c["frame_id"]), [])
        draw(img, boxes, only={"person"})
        d = json.loads(c["detail"] or "{}")
        body.append(card(save(img), f"{c['camera_id']} · {c['check_type']} · điểm <b>{float(c['score']):.3f}</b> · "
                         f"khung {int(c['frame_id']) % 2100} · {html.escape(json.dumps(d, ensure_ascii=False))}"))
    sec.append(("uc4", "UC4 đánh nhau / ngã xe (điểm Cloud)", "Edge gửi đoạn 10 ảnh khi có người đi bộ sát nhau và "
                "quẫy động; Cloud chấm điểm (ngưỡng tạm 0,3). Ảnh: khung cuối của đoạn, box người. Video thử không có "
                "cảnh đánh nhau / ngã xe nên điểm thấp là đúng.", [f'<div class="grid">{"".join(body)}</div>']))

    # events
    body = []
    for e in rows(a.dir, "events"):
        f = os.path.join(a.dir, "media", os.path.basename(e["snapshot_path"] or ""))
        src = save(cv2.imread(f)) if e["snapshot_path"] and os.path.exists(f) else ""
        text = (f"<b>{e['event_type']}</b> · {e['camera_id']} · {e['started_at'][11:19]} → "
                f"{(e['ended_at'] or 'chưa kết thúc')[11:19]} · {html.escape(e['detail'])}")
        body.append(card(src, text) if src else f"<p>{text}</p>")
    sec.append(("events", f"Sự kiện ({len(body)})", "Ùn tắc, tụ tập, lảng vảng, đánh nhau, ngã xe, đông người. Ảnh: "
                "khung lúc sự kiện bắt đầu.", [f'<div class="grid">{"".join(body)}</div>']))

    nav = " · ".join(f'<a href="#{k}">{t.split(":")[0].split("(")[0].strip()}</a>' for k, t, _, _ in sec)
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Xem lại {html.escape(os.path.basename(os.path.abspath(a.dir)))}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}} h2{{margin-top:36px}}
.grid{{display:flex;flex-wrap:wrap;gap:10px}} figure{{margin:0;background:#fff;padding:6px;border-radius:6px;
box-shadow:0 1px 3px #0002;max-width:360px}} figure img{{width:100%;display:block}} figcaption{{font-size:12px;margin-top:4px}}
.pair{{display:flex;gap:4px;background:#fff;padding:4px;border-radius:6px}} .pair figure{{box-shadow:none;max-width:220px}}
table{{border-collapse:collapse;background:#fff}} td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
.note{{color:#555;max-width:900px}} nav{{position:sticky;top:0;background:#f6f6f4;padding:6px 0}}</style></head><body>
<h1>Xem lại lần chạy: {html.escape(os.path.basename(os.path.abspath(a.dir)))}</h1><nav>{nav}</nav>
{"".join(f'<h2 id="{k}">{t}</h2><p class="note">{n}</p>{"".join(b)}' for k, t, n, b in sec)}</body></html>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print(f"{os.path.join(out, 'index.html')}  ({n_img[0]} images)")


if __name__ == "__main__":
    main()
