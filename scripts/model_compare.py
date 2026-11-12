"""Compare detectors on UA-DETRAC from per-frame dumps (vehicles, ignored regions, IoU .5):
confidence threshold chosen on the TRAIN split (max mean per-second F1), then reported on the TEST split:
F1 (mean per second, 15 FPS analytics), F1 by weather, AP50 and AP50:95.

  python -B scripts/model_compare.py ANNOTATION_ROOT name=DUMP_DIR [name=DUMP_DIR ...]
      [--capacity capacity.jsonl] [--out results/detrac/MODEL_COMPARE] [--fixed-threshold T]

--fixed-threshold skips the train sweep (test-split dumps only, e.g. the JPEG quality runs).

Writes OUT.md (report table) and OUT.json. A dump keeps only boxes above its dump threshold, so thresholds
below it are not tried and AP is a lower bound for dumps made at a high threshold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrac  # noqa: E402

THRESHOLDS = [round(x, 2) for x in np.arange(.05, .76, .05)]


def load(root, dump_dir):
    """-> {split: [(annotation dict, dump dict)]} for every dump with an annotation."""
    xmls = detrac._annotations(root)
    out = defaultdict(list)
    for path in sorted(Path(dump_dir).glob("*.json")):
        xml = xmls.get(path.stem)
        if xml is None:
            continue
        out[detrac._split(xml.relative_to(root))].append(
            (detrac.parse_annotation(xml), json.loads(path.read_text(encoding="utf-8"))))
    return out


def f1_rows(pairs, threshold):
    rows = []
    for ann, dump in pairs:
        dump = {**dump, "frames": [{**f, "boxes": [b for b in f["boxes"] if b[1] >= threshold]}
                                   for f in dump["frames"]]}
        rows += detrac.score(ann, dump, analyze_fps=detrac.ANALYTICS_FPS)
    return rows


def dump_floor(pairs):
    confs = [b[1] for _a, d in pairs for f in d["frames"][:500] for b in f["boxes"]]
    return min(confs) if confs else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("models", nargs="+", help="name=dump_dir")
    p.add_argument("--capacity", help="onnx_capacity.py JSON lines tagged with a 'name' field")
    p.add_argument("--out", default="results/detrac/MODEL_COMPARE")
    p.add_argument("--fixed-threshold", type=float, default=None, help="use this threshold instead of the train sweep")
    args = p.parse_args(argv)
    capacity = defaultdict(dict)
    if args.capacity:
        for line in Path(args.capacity).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                capacity[row["name"]][row["batch"]] = row["images_per_s"]
    results = {}
    for spec in args.models:
        name, dump_dir = spec.split("=", 1)
        data = load(args.root, dump_dir)
        floor = dump_floor(data["train"] + data["test"]) - 1e-6
        if args.fixed_threshold is not None:
            best, train = args.fixed_threshold, {args.fixed_threshold: float("nan")}
        else:
            tried = [t for t in THRESHOLDS if t >= floor] or [round(floor, 2)]
            train = {t: float(np.mean([r["f1"] for r in f1_rows(data["train"], t)])) for t in tried}
            best = max(train, key=train.get)
        test = f1_rows(data["test"], best)
        weather = defaultdict(list)
        for r in test:
            weather[r["weather"]].append(r["f1"])
        ap = detrac.average_precision(args.root, dump_dir, split="test")
        results[name] = {"threshold": best, "train_f1": round(train[best], 4),
                         "test_f1": round(float(np.mean([r["f1"] for r in test])), 4),
                         "test_f1_weather": {w: round(float(np.mean(v)), 4) for w, v in sorted(weather.items())},
                         "AP50": round(ap["AP50"], 4), "AP50:95": round(ap["AP50:95"], 4),
                         "dump_floor": round(floor, 3), "test_sequences": len(data["test"]),
                         "capacity_img_s": capacity.get(name, {})}
        print(json.dumps({name: results[name]}), flush=True)
    weathers = sorted({w for r in results.values() for w in r["test_f1_weather"]})
    lines = ["# So sánh model trên UA-DETRAC (tập test)", "",
             "Ngưỡng tin cậy chọn trên tập train (F1 cao nhất), báo trên tập test. F1 = trung bình theo giây ở 15 FPS; "
             "AP class-agnostic cho xe (car/bus/truck), bỏ vùng bị che. Năng lực: ảnh/giây, chỉ tính lời gọi model "
             "trên RTX 4090.", "",
             "| Model | Ngưỡng | F1 test | " + " | ".join(f"F1 {w}" for w in weathers)
             + " | AP50 | AP50:95 | Năng lực b1 | b8 |",
             "|---|---|---|" + "---|" * len(weathers) + "---|---|---|---|"]
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["test_f1"]):
        cap = r["capacity_img_s"]
        lines.append(f"| {name} | {r['threshold']} | {r['test_f1']:.3f} | "
                     + " | ".join(f"{r['test_f1_weather'].get(w, float('nan')):.3f}" for w in weathers)
                     + f" | {r['AP50']:.3f} | {r['AP50:95']:.3f} | {cap.get(1, '-')} | {cap.get(8, '-')} |")
    lines += ["", "Ghi chú: dump lưu sẵn ở một ngưỡng (Cloud cũ 0.10, YOLOv8n 0.25), không thử ngưỡng thấp hơn; "
              "AP của dump ngưỡng cao là cận dưới."]
    Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {args.out}.md / .json")


if __name__ == "__main__":
    main()
