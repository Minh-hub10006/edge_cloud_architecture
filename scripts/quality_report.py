"""Quality table + report from pulled dumps, in one command (UA-DETRAC, docs/DRL_DESIGN_VI.md).

  python -B scripts/quality_report.py ANNOTATION_ROOT RESULTS_DIR

RESULTS_DIR holds index.jsonl (detrac.py index), scene.jsonl and the pulled archives, extracted here:
  quality_247.tgz  -> quality/{yolov8n,yolo11s}          FP32 Edge models, 100 videos (training table)
  quality10_246.tgz -> quality10/{yolov8n,yolo11s}       NPU INT8, 10 test videos (INT8 delta only)
  quality_cloud.tgz -> quality_crf18/..., quality_live/... DETRs on near-lossless and live-transcoded video
Gates: every dump covers every frame of its video (Q3). Writes t_<name>.jsonl tables, table_train.jsonl
(the simulator input: Edge FP32 + Cloud live) and QUALITY_REPORT.md.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrac  # noqa: E402

DUMPS = {  # table name -> (folder under RESULTS_DIR, model)
    "yolov8n_fp32": ("quality/yolov8n", "yolov8n"), "yolo11s_fp32": ("quality/yolo11s", "yolo11s"),
    "yolov8n_int8": ("quality10/yolov8n", "yolov8n"), "yolo11s_int8": ("quality10/yolo11s", "yolo11s"),
    "rtdetrv2_crf18": ("quality_crf18/rtdetrv2", "rtdetrv2"), "lwdetr_crf18": ("quality_crf18/lwdetr", "lwdetr"),
    "rtdetrv2_live": ("quality_live/rtdetrv2", "rtdetrv2"), "lwdetr_live": ("quality_live/lwdetr", "lwdetr"),
}
TRAINING = ("yolov8n_fp32", "yolo11s_fp32", "rtdetrv2_live", "lwdetr_live")
A_MIN_CANDIDATES = (.6, .65, .7, .75, .8, .85)


def extract(results):
    for archive in sorted(results.glob("*.tgz")):
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            if not all((results / n).exists() for n in names if n.endswith(".json")):
                tar.extractall(results, filter="data")


def verify(results, frames):
    bad = {}
    for name, (folder, _model) in DUMPS.items():
        for path in sorted((results / folder).glob("*.json")):
            n = len(json.loads(path.read_text(encoding="utf-8"))["frames"])
            if n != frames.get(path.stem):
                bad.setdefault(name, []).append(f"{path.stem} {n}/{frames.get(path.stem)}")
    return bad


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    annotations, results = Path(argv[0]), Path(argv[1])
    extract(results)
    frames = {r["sequence"]: r["frames"] for r in map(json.loads, (results / "index.jsonl").read_text().splitlines())
              if "sequence" in r}
    bad = verify(results, frames)
    if bad:
        raise SystemExit(f"[Q3] FAIL incomplete dumps: {bad}")
    print("[Q3] PASS every dump covers every frame")
    tables = {}
    for name, (folder, model) in DUMPS.items():
        if not (results / folder).is_dir():
            print(f"[Q4] skip {name}: {folder} not pulled yet")
            continue
        rows = list(detrac.table(annotations, results / folder, model))
        with open(results / f"t_{name}.jsonl", "w", encoding="utf-8") as out:
            out.writelines(json.dumps(r) + "\n" for r in rows)
        tables[name] = rows
        print(f"[Q4] {name}: {len(rows)} seconds")
    key = lambda r: (r["sequence"], r["window"])  # noqa: E731
    lines = ["# UA-DETRAC per-second quality report", "",
             "Per-second F1 (vehicles, IoU 0.5, ignored regions), live confidence thresholds.", "",
             "## Mean F1 by split / weather", "", "| table | " + " | ".join(
                 f"{s}/{w}" for s in ("train", "test") for w in ("sunny", "cloudy", "rainy", "night")) + " | train | test |",
             "|---" * 11 + "|"]
    for name, rows in tables.items():
        cell = lambda s, w=None: mean([r["f1"] for r in rows if r["split"] == s and (w is None or r["weather"] == w)])  # noqa: E731
        lines.append(f"| {name} | " + " | ".join(f"{cell(s, w):.3f}" for s in ("train", "test")
                                                  for w in ("sunny", "cloudy", "rainy", "night"))
                     + f" | {cell('train'):.3f} | {cell('test'):.3f} |")
    deltas = {}
    lines += ["", "## Paired deltas (same seconds)", "", "| comparison | n | mean | median | p10 | p90 |", "|---|---|---|---|---|---|"]
    for label, a, b in (("yolov8n INT8 - FP32", "yolov8n_int8", "yolov8n_fp32"),
                        ("yolo11s INT8 - FP32", "yolo11s_int8", "yolo11s_fp32"),
                        ("rtdetrv2 live - crf18", "rtdetrv2_live", "rtdetrv2_crf18"),
                        ("lwdetr live - crf18", "lwdetr_live", "lwdetr_crf18")):
        if a in tables and b in tables:
            ref = {key(r): r["f1"] for r in tables[b]}
            d = np.array([r["f1"] - ref[key(r)] for r in tables[a] if key(r) in ref])
            deltas[label] = float(d.mean())
            lines.append(f"| {label} | {len(d)} | {d.mean():+.4f} | {np.median(d):+.4f} | "
                         f"{np.percentile(d, 10):+.3f} | {np.percentile(d, 90):+.3f} |")
    present = [t for t in TRAINING if t in tables]
    if len(present) == len(TRAINING):
        per = {}
        for name in TRAINING:
            for r in tables[name]:
                if r["split"] == "train":
                    per.setdefault(key(r), {})[name] = r["f1"]
        seconds = [v for v in per.values() if len(v) == len(TRAINING)]
        lines += ["", "## Choosing A_min (train seconds below the floor)", "",
                  "| A_min | " + " | ".join(TRAINING) + " | best of 4 (oracle) |", "|---" * (len(TRAINING) + 2) + "|"]
        for a_min in A_MIN_CANDIDATES:
            shares = [mean([s[t] < a_min for s in seconds]) for t in TRAINING]
            oracle = mean([max(s.values()) < a_min for s in seconds])
            lines.append(f"| {a_min} | " + " | ".join(f"{x:.1%}" for x in shares) + f" | {oracle:.1%} |")
        with open(results / "table_train.jsonl", "w", encoding="utf-8") as out:
            for name in TRAINING:
                out.writelines(json.dumps(r) + "\n" for r in tables[name])
        print(f"[Q5] simulator table -> {results / 'table_train.jsonl'}")
    scene = {key(r): r for r in map(json.loads, (results / "scene.jsonl").read_text().splitlines())}
    lines += ["", "## Spearman: label-free features vs F1 (train)", "",
              "| table | vehicle_count | box_change | small_ratio | brightness | motion |", "|---|---|---|---|---|---|"]
    for name in present:
        rows = [r for r in tables[name] if r["split"] == "train" and r.get("box_change") is not None
                and r.get("small_ratio") is not None and key(r) in scene]
        f1 = np.array([r["f1"] for r in rows])
        own = [spearman(np.array([r[f] for r in rows]), f1) for f in ("vehicle_count", "box_change", "small_ratio")]
        pix = [spearman(np.array([scene[key(r)][f] for r in rows]), f1) for f in ("brightness", "motion")]
        lines.append(f"| {name} | " + " | ".join(f"{x:+.3f}" for x in own + pix) + " |")
    (results / "QUALITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results / "deltas.json").write_text(json.dumps(deltas, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
