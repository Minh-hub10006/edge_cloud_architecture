"""Gate G3 (plan 4/11): F1 against UA-DETRAC test labels for path B (NPU head + Cloud tail) vs path C (JPEG + full).

  prep:  python -B scripts/cut_g3_eval.py prep DETRAC_ROOT OUT_DIR [--per-seq 5]
  prep-vn: python -B scripts/cut_g3_eval.py prep-vn VIDEO_DIR LABEL_DIR OUT_DIR [--per-seq 40]
         VN-Hard test videos + DEIMv2-X auto labels (results/vn_hard/labels_auto/test_<video>.jsonl, conf >= .5,
         car/motorcycle/bus/truck: VN traffic is mostly motorcycles)
         -> OUT_DIR/test_raw/*.raw (NHWC float32 640, INTER_AREA, RGB, /255 = cut_head_raw.py) + manifest.json
  score: python -B scripts/cut_g3_eval.py score OUT_DIR FULL.onnx HEAD.onnx TAIL.onnx NPU_LIST NPU_OUT [NPU_OUT ...]

Frames: `per-seq` labeled frames spread evenly over each of the 40 test sequences. Variants, all RT-DETRv2-S:
  ref = full model on the uncompressed 640 frame; C = full model on the JPEG q85 640 frame (edge_agent
  encode_cloud_jpeg); B-float = float head + tail (= ref, sanity); B-<npu> = NPU head output (snpe-net-run) + tail,
  with the cut tensor sent as uint8 + zstd (per-tensor range) as on the wire; *_2bit = the same through CUT2 2-bit
  (contracts/cut_tensor.py pack/unpack, the live default since 14/11), mean wire size printed per variant.
Scoring = scripts/detrac.py: vehicles = COCO car/bus/truck, score >= .6 (MODEL_COMPARE), ignored regions dropped,
greedy IoU .5; micro F1 over all frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detrac import index, match_frame, parse_annotation  # noqa: E402

VEHICLES = {2, 5, 7}  # COCO contiguous car, bus, truck (DETRAC labels no motorcycles)
THR = .6
S = 640


def prep(root, out, per_seq):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    (out / "test_raw").mkdir(parents=True)
    rows = []
    for seq in index(root):
        if seq["split"] != "test" or not seq["annotation"]:
            continue
        labeled = sorted(parse_annotation(seq["annotation"])["frames"])
        for num in labeled[:: max(1, len(labeled) // per_seq)][:per_seq]:
            bgr = cv2.imread(str(Path(seq["images"]) / f"img{num:05d}.jpg"))
            x = cv2.cvtColor(cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            name = f"{seq['sequence']}_{num:05d}.raw"
            (x.astype(np.float32) / 255).tofile(out / "test_raw" / name)
            rows.append({"raw": name, "sequence": seq["sequence"], "frame": num, "annotation": seq["annotation"],
                         "image": str(Path(seq["images"]) / f"img{num:05d}.jpg"), "w": bgr.shape[1], "h": bgr.shape[0]})
    (out / "manifest.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} frames from {len({r['sequence'] for r in rows})} test sequences -> {out}")


VN_CLASSES = {2, 3, 5, 7}
VN_LABEL_CONF = .5


def _raw(out, name, bgr):
    x = cv2.cvtColor(cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    (x.astype(np.float32) / 255).tofile(out / "test_raw" / name)


def prep_vn(video_dir, label_dir, out, per_seq):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    (out / "test_raw").mkdir(parents=True)
    rows = []
    for video in sorted(Path(video_dir).glob("*.mp4")):
        labels = {}
        for line in open(Path(label_dir) / f"test_{video.stem}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            labels[r["frame"]] = r["boxes"]
        nums = sorted(labels)
        cap = cv2.VideoCapture(str(video))
        for num in nums[:: max(1, len(nums) // per_seq)][:per_seq]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, num)
            ok, bgr = cap.read()
            assert ok, (video, num)
            h, w = bgr.shape[:2]
            gt = [[x1 * w, y1 * h, x2 * w, y2 * h] for c, conf, x1, y1, x2, y2 in labels[num]
                  if conf >= VN_LABEL_CONF and int(c) in VN_CLASSES]
            name = f"{video.stem}_{num:05d}.raw"
            _raw(out, name, bgr)
            rows.append({"raw": name, "sequence": video.stem, "frame": num, "gt": gt, "classes": sorted(VN_CLASSES),
                         "w": w, "h": h})
    (out / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"{len(rows)} frames from {len({r['sequence'] for r in rows})} VN test videos -> {out}")


def _vehicles(out, names, classes=VEHICLES):
    o = dict(zip(names, out))
    s, b, lab = o["scores"][0], o["boxes"][0], o["labels"][0]
    return [(float(c), tuple(map(float, bb))) for c, bb, l in zip(s, b, lab) if c >= THR and int(l) in classes]


def _u8_wire(t):
    lo, hi = float(t.min()), float(t.max())
    q = np.round((t - lo) / max(hi - lo, 1e-12) * 255).astype(np.uint8)
    return (q.astype(np.float32) / 255 * (hi - lo) + lo).astype(np.float32)


def _cut2_wire(t, sizes):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contracts.cut_tensor import pack, unpack
    data = pack(np.ascontiguousarray(t, dtype=np.float32), width=int(sizes[0, 0]), height=int(sizes[0, 1]),
                head_id="g3", bits=2)
    return unpack(data, expected_head_id="g3")[0].astype(np.float32), len(data)


def score(out, full_p, head_p, tail_p, npu_list, npu_outs):
    import onnxruntime as ort
    out = Path(out)
    so = {"providers": ["CPUExecutionProvider"]}
    full, head, tail = (ort.InferenceSession(p, **so) for p in (full_p, head_p, tail_p))
    names = [o.name for o in full.get_outputs()]
    cut = head.get_outputs()[0].name
    manifest = {r["raw"]: r for r in json.loads((out / "manifest.json").read_text(encoding="utf-8"))}
    order = [Path(line.strip()).name for line in open(npu_list) if line.strip()]
    assert set(order) == set(manifest), "NPU test list and manifest differ"
    gt_cache, tot, wire = {}, {}, {}
    for i, name in enumerate(order):
        r = manifest[name]
        if "gt" in r:  # VN: auto labels carried in the manifest, no ignored regions
            ann, gt = {"ignored": []}, [tuple(g) for g in r["gt"]]
        else:
            if r["annotation"] not in gt_cache:
                gt_cache[r["annotation"]] = parse_annotation(r["annotation"])
            ann = gt_cache[r["annotation"]]
            gt = ann["frames"][r["frame"]]
        sizes = np.array([[r["w"], r["h"]]], dtype=np.int64)
        x = np.fromfile(out / "test_raw" / name, dtype=np.float32).reshape(1, S, S, 3)
        bgr = cv2.cvtColor((x[0] * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        xj = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        xn = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        mid = head.run(None, {"images": xn})[0]
        preds = {"ref": full.run(None, {"images": xn, "orig_target_sizes": sizes}),
                 "C_jpeg85": full.run(None, {"images": xj.transpose(2, 0, 1)[None], "orig_target_sizes": sizes}),
                 "B_float_u8": tail.run(None, {cut: _u8_wire(mid), "orig_target_sizes": sizes})}
        two, n = _cut2_wire(mid, sizes)
        preds["B_float_2bit"] = tail.run(None, {cut: two, "orig_target_sizes": sizes})
        wire.setdefault("B_float_2bit", []).append(n)
        for d in npu_outs:
            (f,) = list((Path(d) / f"Result_{i}").rglob("*.raw"))
            npu = np.fromfile(f, dtype=np.float32).reshape(1, mid.shape[2], mid.shape[3], mid.shape[1])
            npu = np.ascontiguousarray(npu.transpose(0, 3, 1, 2))
            preds[f"B_{Path(d).name.removeprefix('npu_out_')}_u8"] = tail.run(
                None, {cut: _u8_wire(npu), "orig_target_sizes": sizes})
            two, n = _cut2_wire(npu, sizes)
            key = f"B_{Path(d).name.removeprefix('npu_out_')}_2bit"
            preds[key] = tail.run(None, {cut: two, "orig_target_sizes": sizes})
            wire.setdefault(key, []).append(n)
        for k, p in preds.items():
            tp, fp, fn = match_frame(gt, _vehicles(p, names, set(r.get("classes", VEHICLES))), ann["ignored"])
            t = tot.setdefault(k, [0, 0, 0])
            t[0] += tp; t[1] += fp; t[2] += fn  # noqa: E702
    print(f"{len(order)} frames | vehicles score >= {THR}, IoU .5, micro F1")
    rows = []
    for k, (tp, fp, fn) in tot.items():
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rows.append({"variant": k, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
                     "recall": tp / max(tp + fn, 1), "f1": f1,
                     "wire_kb": round(float(np.mean(wire[k])) / 1024, 1) if k in wire else None})
        kb = f"  {rows[-1]['wire_kb']} KB" if k in wire else ""
        print(f"  {k:22s} F1 {f1:.4f}  P {rows[-1]['precision']:.4f}  R {rows[-1]['recall']:.4f}  (tp {tp} fp {fp} fn {fn}){kb}")
    (out / "g3_result.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("root")
    a.add_argument("out")
    a.add_argument("--per-seq", type=int, default=5)
    v = sub.add_parser("prep-vn")
    for k in ("video_dir", "label_dir", "out"):
        v.add_argument(k)
    v.add_argument("--per-seq", type=int, default=40)
    b = sub.add_parser("score")
    for k in ("out", "full", "head", "tail", "npu_list"):
        b.add_argument(k)
    b.add_argument("npu_outs", nargs="*")
    x = p.parse_args()
    if x.cmd == "prep":
        prep(x.root, x.out, x.per_seq)
    elif x.cmd == "prep-vn":
        prep_vn(x.video_dir, x.label_dir, x.out, x.per_seq)
    else:
        score(x.out, x.full, x.head, x.tail, x.npu_list, x.npu_outs)


if __name__ == "__main__":
    main()
