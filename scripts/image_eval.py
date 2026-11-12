"""Edge vs Cloud gap and probe signal on a still-image traffic set (COCO-format labels, e.g. BMD-45).

  python -B scripts/image_eval.py dump IMAGE_DIR COCO_JSON OUT_DIR --yolo Y11.onnx --rtdetr RT.onnx --dfinex DX.onnx
  python -B scripts/image_eval.py score OUT_DIR COCO_JSON [--out results/bmd45/IMAGE_EVAL]

dump: every image through the deployed YOLO11s float ONNX (yolo11s_contract prepare/decode, as Edge 247), RT-DETRv2-S
and D-FINE-X (detr_candidate_dump preprocessing), CPU onnxruntime; boxes >= 0.05 kept, normalized, one JSONL row per
image (resumable). score: class-agnostic vehicles (labels: every class but Bicycle; detections: COCO car, motorcycle,
bus, truck), IoU 0.5, MODEL_COMPARE thresholds; per-image F1, gE = F1(D-FINE-X) - F1(YOLO11s), probe = vehicles
D-FINE-X finds and YOLO11s misses; AUC and the gain of putting 30% of images on the Cloud by oracle / probe / random.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
DET_VEHICLES = {2, 3, 5, 7}          # COCO contiguous car, motorcycle, bus, truck
THRESHOLDS = {"yolo11s": .25, "rtdetrv2": .6, "dfine_x": .65}
PROBE_CONF = {"yolo11s": .30, "dfine_x": .65}


def dump(image_dir, coco_json, out_dir, *, yolo, rtdetr, dfinex):
    import cv2
    import onnxruntime as ort
    import yolo11s_contract as yc
    from detr_candidate_dump import preprocess
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    out = Path(out_dir) / "dumps.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["image_id"] for x in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    cpu = ["CPUExecutionProvider"]
    y = ort.InferenceSession(yolo, providers=cpu)
    detr = {"rtdetrv2": ort.InferenceSession(rtdetr, providers=cpu), "dfine_x": ort.InferenceSession(dfinex, providers=cpu)}
    with out.open("a", encoding="utf-8") as sink:
        for n, image in enumerate(coco["images"]):
            if image["id"] in done:
                continue
            bgr = cv2.imread(str(Path(image_dir) / image["file_name"].replace("/", "_")))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            g = yc.prepare(bgr)
            raw = dict(zip([o.name for o in y.get_outputs()], y.run(None, {y.get_inputs()[0].name: g.tensor})))
            shaped = {k: np.asarray(raw[k], np.float32).reshape(s) for k, s in yc.SHAPES.items()}
            row = {"image_id": image["id"], "yolo11s": [[d.class_id, round(d.confidence, 4), d.left, d.top, d.right, d.bottom]
                                                        for d in yc.decode(shaped, g, nms_iou=.5, confidence=.05)]}
            for name, session in detr.items():
                labels, boxes, scores = session.run(["labels", "boxes", "scores"], {
                    "images": preprocess([bgr], "none"), "orig_target_sizes": np.array([[w, h]], np.int64)})
                keep = scores[0] >= .05
                row[name] = [[int(l), round(float(s), 4), float(b[0]) / w, float(b[1]) / h, float(b[2]) / w, float(b[3]) / h]
                             for l, b, s in zip(labels[0][keep], boxes[0][keep], scores[0][keep])]
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if n % 100 == 0:
                print(f"dumped {n + 1}/{len(coco['images'])}", flush=True)


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(refs, preds, iou=.5):
    """Greedy (highest score first) one-to-one matching -> number of matched refs."""
    used, hits = set(), 0
    for p in sorted(preds, key=lambda p: -p[0]):
        best, best_k = 0.0, None
        for k, r in enumerate(refs):
            if k not in used:
                v = _iou(r, p[1])
                if v > best:
                    best, best_k = v, k
        if best >= iou:
            used.add(best_k)
            hits += 1
    return hits


def f1(gt, preds):
    tp = match(gt, preds)
    return 1.0 if not gt and not preds else 2 * tp / (len(gt) + len(preds))


def detections(row, model, threshold):
    return [(b[1], b[2:6]) for b in row[model] if b[0] in DET_VEHICLES and b[1] >= threshold]


def score(out_dir, coco_json, out):
    from sklearn.metrics import roc_auc_score
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    bicycle = {c["id"] for c in coco["categories"] if c["name"].lower() == "bicycle"}
    size = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    gt = {}
    for a in coco["annotations"]:
        if a["category_id"] not in bicycle:
            w, h = size[a["image_id"]]
            x, y, bw, bh = a["bbox"]
            gt.setdefault(a["image_id"], []).append([x / w, y / h, (x + bw) / w, (y + bh) / h])
    rows = []
    for line in (Path(out_dir) / "dumps.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        g = gt.get(r["image_id"], [])
        item = {"image_id": r["image_id"], "gt": len(g)}
        for model, thr in THRESHOLDS.items():
            item[f"f1_{model}"] = f1(g, detections(r, model, thr))
        ref = [p[1] for p in detections(r, "dfine_x", PROBE_CONF["dfine_x"])]
        item["probe_missed"] = len(ref) - match(ref, detections(r, "yolo11s", PROBE_CONF["yolo11s"]))
        item["gE"] = item["f1_dfine_x"] - item["f1_yolo11s"]
        item["gC"] = item["f1_dfine_x"] - item["f1_rtdetrv2"]
        rows.append(item)
    g, p = np.array([r["gE"] for r in rows]), np.array([r["probe_missed"] for r in rows], float)
    k = int(.3 * len(g))
    gain = lambda s: float(g[np.argsort(-s, kind="stable")[:k]].sum() / len(g))  # noqa: E731
    summary = {"images": len(rows), "mean_f1": {m: round(float(np.mean([r[f"f1_{m}"] for r in rows])), 4) for m in THRESHOLDS},
               "gE": round(float(g.mean()), 4), "gC": round(float(np.mean([r["gC"] for r in rows])), 4),
               "hard_share": round(float(np.mean(g >= .10)), 3),
               "probe_auc": round(float(roc_auc_score(g >= .10, p)), 3) if 0 < np.mean(g >= .10) < 1 else None,
               "cloud30_gain": {"oracle": round(gain(g), 4), "probe": round(gain(p), 4), "random": round(.3 * float(g.mean()), 4)}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out + ".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    Path(out + ".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("image_dir")
    d.add_argument("coco_json")
    d.add_argument("out_dir")
    d.add_argument("--yolo", required=True)
    d.add_argument("--rtdetr", required=True)
    d.add_argument("--dfinex", required=True)
    s = sub.add_parser("score")
    s.add_argument("out_dir")
    s.add_argument("coco_json")
    s.add_argument("--out", default="results/bmd45/IMAGE_EVAL")
    args = p.parse_args(argv)
    if args.command == "dump":
        dump(args.image_dir, args.coco_json, args.out_dir, yolo=args.yolo, rtdetr=args.rtdetr, dfinex=args.dfinex)
    else:
        score(args.out_dir, args.coco_json, args.out)


if __name__ == "__main__":
    main()
