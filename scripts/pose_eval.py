"""UC4 pose model choice (P3, 11/11): YOLO11-pose (Ultralytics, AGPL-3.0 like the YOLO11s already used) vs RTMO
(OpenMMLab, Apache-2.0), both as ONNX on ONNX Runtime so the comparison is fair.

  python -B scripts/pose_eval.py coco ANN.json IMG_DIR MODEL [MODEL ...] [--limit 500]
  python -B scripts/pose_eval.py show MODEL IMAGE [IMAGE ...] --out DIR
  python -B scripts/pose_eval.py triton RTMO.onnx IMAGE [IMAGE ...] [--model pose_rtmo_s]   # on 249

MODEL: a YOLO-pose ONNX (output 1x56xN: cx, cy, w, h, score, 17 x (x, y, v)) or an RTMO ONNX from OpenMMLab's onnx_sdk
(outputs dets 1xNx5, keypoints 1xNx17x3; letterbox with 114, as rtmlib does).
coco: COCO val2017 person keypoints, the first `limit` images (by id) that have people with keypoints; images missing in
      IMG_DIR are downloaded from images.cocodataset.org. -> keypoint AP / AP50 / AP-medium / AP-large / AR (pycocotools
      OKS) and CPU ms per image.
show: draws skeletons on images (our VN frames: riders on motorbikes, seen from above).
triton: (1) parity - the Triton model vs the same ONNX on CPU, per image: people found, keypoint distance in pixels;
        (2) speed - 1, 2, 4, 8 concurrent callers for --seconds each: images/s and latency p50/p95 (Triton batches).
11/11 VN frames 1080x1984: the whole frame at 640 finds few far people (RTMO-s 12 -> 28 on the top half alone), so the
Cloud runs pose on the region around the triggering people, not on the whole frame.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


class PoseModel:
    """-> list of (score, box xyxy, keypoints 17x3 [x, y, score]) in image pixels."""

    def __init__(self, path, conf=.25, iou=.65, triton=None):
        self.path, self.conf, self.iou = Path(path), conf, iou
        self.name = self.path.stem
        if triton:   # RTMO served by Triton (inference.triton_detr.TritonSession: run(names, feed))
            from inference.triton_detr import TritonSession
            self.sess, self.rtmo, self.input, self.name = TritonSession(triton), True, "input", triton
            return
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.rtmo = len(self.sess.get_outputs()) == 2
        self.input = self.sess.get_inputs()[0].name

    def letterbox(self, bgr, size=640, pad=114):
        h, w = bgr.shape[:2]
        r = min(size / h, size / w)
        img = np.full((size, size, 3), pad, np.uint8)
        img[:int(h * r), :int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        return img, r

    def __call__(self, bgr):
        img, r = self.letterbox(bgr)
        if self.rtmo:   # the production path (inference/pose_rtmo.py), so what is measured is what runs
            from inference.pose_rtmo import letterbox, people
            x, r = letterbox(bgr)
            dets, kps = self.sess.run(["dets", "keypoints"], {self.input: x})
            return people(dets[0], kps[0], r, self.conf, self.iou)
        else:           # YOLO-pose: RGB / 255, raw head -> NMS here
            x = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
            out = self.sess.run(None, {self.input: x})[0][0].T   # N x 56
            out = out[out[:, 4] >= self.conf]
            cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) / r
            scores, kp = out[:, 4], out[:, 5:].reshape(-1, 17, 3).copy()
            kp[..., :2] /= r
        keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes],
                                scores.tolist(), self.conf, self.iou) if len(boxes) else []
        keep = np.array(keep).reshape(-1)
        return [(float(scores[i]), boxes[i], kp[i]) for i in keep if scores[i] >= self.conf]


def coco_eval(ann, img_dir, models, limit):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(ann)
    person = gt.getCatIds(catNms=["person"])
    ids = sorted({a["image_id"] for a in gt.loadAnns(gt.getAnnIds(catIds=person, iscrowd=False)) if a["num_keypoints"] > 0})
    ids = ids[:limit]
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    for i in ids:
        f = Path(img_dir, gt.loadImgs(i)[0]["file_name"])
        if not f.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{f.name}", f)
    print(f"{len(ids)} COCO val2017 images with people, "
          f"{len(gt.getAnnIds(imgIds=ids, catIds=person, iscrowd=False))} people", flush=True)
    for m in models:
        net, res, ms = PoseModel(m, conf=.01), [], []
        for i in ids:
            bgr = cv2.imread(str(Path(img_dir, gt.loadImgs(i)[0]["file_name"])))
            t = time.perf_counter()
            people = net(bgr)
            ms.append((time.perf_counter() - t) * 1000)
            for score, box, kp in people:
                res.append({"image_id": i, "category_id": person[0], "score": score,
                            "keypoints": [float(v) for xyv in kp for v in (xyv[0], xyv[1], 1.0)]})
        ev = COCOeval(gt, gt.loadRes(res) if res else COCO(), "keypoints")
        ev.params.imgIds = ids
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        print(f"  {net.name:45s} AP {s[0]:.3f}  AP50 {s[1]:.3f}  AP-medium {s[3]:.3f}  AP-large {s[4]:.3f}  AR {s[5]:.3f}"
              f"  CPU {np.median(ms):.0f} ms/img", flush=True)


def show(model, images, out):
    net = PoseModel(model, conf=.3)
    Path(out).mkdir(parents=True, exist_ok=True)
    for p in images:
        bgr = cv2.imread(str(p))
        people = net(bgr)
        for score, box, kp in people:
            for a, b in SKELETON:
                if kp[a, 2] > .3 and kp[b, 2] > .3:
                    cv2.line(bgr, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (0, 255, 0), 2)
            for x, y, v in kp:
                if v > .3:
                    cv2.circle(bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(bgr, f"{net.name[:20]}: {len(people)} people", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        cv2.imwrite(str(Path(out, f"{Path(p).stem}_{net.name[:12]}.jpg")), bgr)
        print(f"{Path(p).name}: {len(people)} people")


def triton_check(onnx, images, model, seconds):
    import statistics
    from concurrent.futures import ThreadPoolExecutor
    local, remote = PoseModel(onnx, conf=.3), PoseModel(onnx, conf=.3, triton=model)
    frames = [cv2.imread(str(p)) for p in images]
    worst = 0.0
    for p, bgr in zip(images, frames):
        a, b = local(bgr), remote(bgr)
        # 11/11 on 249: pairing by score swapped two near-equal people (437 px); pair by nearest box, visible joints only
        d = []
        for _, box, kp in a:
            if b:
                _, _, other = min(b, key=lambda r: float(np.abs(r[1] - box).sum()))
                seen = (kp[:, 2] > .3) & (other[:, 2] > .3)
                d.append(float(np.abs(kp[seen, :2] - other[seen, :2]).max()) if seen.any() else 0.0)
        worst = max([worst] + d)
        print(f"  {Path(p).name}: CPU {len(a)} people, Triton {len(b)}, keypoints differ <= {max(d, default=0):.1f} px")
    print(f"parity: {'PASS' if worst <= 2 else 'CHECK'} (worst {worst:.1f} px)", flush=True)
    for callers in (1, 2, 4, 8):
        lat, stop = [], time.perf_counter() + seconds

        def worker(k):
            i = k
            while time.perf_counter() < stop:
                t = time.perf_counter()
                remote(frames[i % len(frames)])
                lat.append((time.perf_counter() - t) * 1000)
                i += 1
        with ThreadPoolExecutor(callers) as ex:
            list(ex.map(worker, range(callers)))
        q = statistics.quantiles(lat, n=20)
        print(f"  {callers} callers: {len(lat) / seconds:6.1f} images/s  p50 {statistics.median(lat):6.1f} ms"
              f"  p95 {q[18]:6.1f} ms  (includes letterbox + decode on CPU)", flush=True)
    remote.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coco", "show", "triton"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="pose_show")
    ap.add_argument("--model", default="pose_rtmo_s")
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    if a.cmd == "coco":
        coco_eval(a.args[0], a.args[1], a.args[2:], a.limit)
    elif a.cmd == "triton":
        triton_check(a.args[0], a.args[1:], a.model, a.seconds)
    else:
        show(a.args[0], a.args[1:], a.out)


if __name__ == "__main__":
    sys.exit(main())
