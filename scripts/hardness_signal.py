"""L1 check: can an Edge tell a hard second (D-FINE-X beats YOLO11s by >= .10 F1) from what it measures itself?

  python -B scripts/hardness_signal.py ANNOTATION_ROOT VIDEO_DIR [--table results/detrac/hardness.jsonl]
      [--yolo results/detrac/quality/yolo11s] [--out results/detrac/HARDNESS_SIGNAL.md]

Per second (the hardness.jsonl grid) the Edge's own YOLO11s output gives confidence / box statistics, and one
decoded frame gives brightness / contrast / blur. A classifier trained on the train split is scored on the
test split (AUC). Gate: AUC >= 0.75, otherwise the state cannot see the scenes where Cloud pays off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FPS = 25
FEATURES = ("conf_mean", "count", "small_ratio", "uncertain_ratio", "brightness", "contrast", "blur")


def box_features(frames, threshold=.25):
    """YOLO dump frames of one second -> confidence and box statistics (normalized boxes, 540 px tall source)."""
    confs, counts, small, uncertain = [], [], [], []
    for f in frames:
        kept = [b for b in f["boxes"] if b[1] >= threshold]
        counts.append(len(kept))
        confs += [b[1] for b in kept]
        small += [(b[5] - b[3]) * 540 < 32 for b in kept]
        uncertain += [b[1] < .5 for b in kept]
    mean = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731
    return {"conf_mean": mean(confs), "count": mean(counts), "small_ratio": mean(small),
            "uncertain_ratio": mean(uncertain)}


def image_features(gray):
    import cv2
    return {"brightness": float(gray.mean()) / 255, "contrast": float(gray.std()) / 255,
            "blur": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))}


def rows(table, yolo_dir, video_dir):
    import cv2
    by_seq = {}
    for r in table:
        by_seq.setdefault(r["sequence"], {})[r["second"]] = r
    out = []
    for seq, seconds in sorted(by_seq.items()):
        dump = json.loads((Path(yolo_dir) / f"{seq}.json").read_text(encoding="utf-8"))
        frames = {}
        for f in dump["frames"]:
            frames.setdefault(f["frame"] // FPS, []).append(f)
        capture, index = cv2.VideoCapture(str(Path(video_dir) / f"{seq}.mp4")), 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            second = index // FPS
            if index % FPS == FPS // 2 and second in seconds and second in frames:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                r = seconds[second]
                out.append({"split": r["split"], "sequence": seq, "second": second, "weather": r["weather"],
                            "hard": r["gE"] >= .10, "gE": r["gE"],
                            **box_features(frames[second]), **image_features(gray)})
            index += 1
        capture.release()
    return out


def probe_table(table, yolo_dir, ref_dir, out, *, yolo_conf=.30, ref_conf=.65, offsets=(6, 18)):
    """Per second: vehicles the reference (D-FINE-X) finds and the Edge YOLO misses, on the frames a live probe
    could pick (two per second) -> JSONL rows the simulator replays (docs/PLAN_PROBE_VI.md P5)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import detrac

    def boxes(folder, seq, thr):
        frames = json.loads((Path(folder) / f"{seq}.json").read_text(encoding="utf-8"))["frames"]
        return {f["frame"]: [b[2:6] for b in f["boxes"] if b[1] >= thr and int(b[0]) in detrac.VEHICLE_CLASSES]
                for f in frames}

    def missed(edge, ref):
        used, count = set(), 0
        for r in ref:
            best, best_k = 0.0, None
            for k, e in enumerate(edge):
                if k in used:
                    continue
                w = max(0.0, min(r[2], e[2]) - max(r[0], e[0]))
                h = max(0.0, min(r[3], e[3]) - max(r[1], e[1]))
                inter = w * h
                union = (r[2] - r[0]) * (r[3] - r[1]) + (e[2] - e[0]) * (e[3] - e[1]) - inter
                if union > 0 and inter / union > best:
                    best, best_k = inter / union, k
            if best >= .5:
                used.add(best_k)
            else:
                count += 1
        return count

    rows, cache = [], {}
    for r in table:
        seq = r["sequence"]
        if seq not in cache:
            cache[seq] = boxes(yolo_dir, seq, yolo_conf), boxes(ref_dir, seq, ref_conf)
        edge, ref = cache[seq]
        frames = [r["second"] * FPS + o for o in offsets]
        rows.append({"split": r["split"], "sequence": seq, "second": r["second"],
                     "probe_missed": [missed(edge.get(f, []), ref.get(f, [])) for f in frames]})
    Path(out).write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("video_dir")
    p.add_argument("--table", default="results/detrac/hardness.jsonl")
    p.add_argument("--yolo", default="results/detrac/quality/yolo11s")
    p.add_argument("--out", default="results/detrac/HARDNESS_SIGNAL.md")
    args = p.parse_args(argv)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    table = [json.loads(x) for x in Path(args.table).read_text(encoding="utf-8").splitlines()]
    data = rows(table, args.yolo, args.video_dir)
    Path(args.out).with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")
    split = {s: [r for r in data if r["split"] == s] for s in ("train", "test")}
    X = {s: np.array([[r[f] for f in FEATURES] for r in v]) for s, v in split.items()}
    y = {s: np.array([r["hard"] for r in v]) for s, v in split.items()}
    lines = ["# L1: Edge có nhận ra giây khó không (gE >= 0.10)", "",
             f"Giây: train {len(y['train'])} (khó {y['train'].mean():.0%}), test {len(y['test'])} "
             f"(khó {y['test'].mean():.0%}). Đặc trưng: {', '.join(FEATURES)}.", "",
             "| Bộ phân loại / đặc trưng | AUC test |", "|---|---|"]
    models = {"Logistic (tất cả)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "Gradient boosting (tất cả)": GradientBoostingClassifier(random_state=0)}
    results = {}
    for name, model in models.items():
        model.fit(X["train"], y["train"])
        results[name] = roc_auc_score(y["test"], model.predict_proba(X["test"])[:, 1])
        lines.append(f"| {name} | {results[name]:.3f} |")
    for i, f in enumerate(FEATURES):  # one feature alone, sign chosen on train
        sign = 1 if roc_auc_score(y["train"], X["train"][:, i]) >= .5 else -1
        lines.append(f"| chỉ {f} | {roc_auc_score(y['test'], sign * X['test'][:, i]):.3f} |")
    best = max(results.values())
    lines += ["", f"Cổng AUC >= 0.75: **{'ĐẠT' if best >= .75 else 'KHÔNG ĐẠT'}** (tốt nhất {best:.3f})."]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252
    print("\n".join(lines))


if __name__ == "__main__":
    main()
