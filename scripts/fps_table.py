"""Per-second F1 of every model at each analytics FPS (shown frames at 15 FPS, boxes held between analyses).

  python -B scripts/fps_table.py ANNOTATION_ROOT RESULTS_DIR [--fps 5 10 15] [--out table_fps.jsonl]

Same dumps as quality_report.py's simulator table: Edge models FP32 (quality/), Cloud models live-transcoded
(quality_live/). Rows carry an "fps" field; one process per (model, FPS).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrac  # noqa: E402

DUMPS = {"yolov8n": "quality/yolov8n", "yolo11s": "quality/yolo11s",
         "rtdetrv2": "quality_live/rtdetrv2", "lwdetr": "quality_live/lwdetr"}


def job(args):
    root, results, model, fps, display = args
    rows = list(detrac.table(root, Path(results) / DUMPS[model], model, analyze_fps=fps, display_fps=display))
    return [{**r, "display_fps": display} for r in rows]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("results")
    p.add_argument("--fps", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--out", default="table_fps.jsonl")
    p.add_argument("--display", type=int, default=15, help="shown frames per second (media stream FPS)")
    args = p.parse_args(argv)
    jobs = [(args.root, args.results, m, f, args.display) for m in DUMPS for f in args.fps if f <= args.display]
    with Pool(min(len(jobs), 12)) as pool:
        parts = pool.map(job, jobs)
    out = Path(args.results) / args.out
    with open(out, "w", encoding="utf-8") as f:
        for (_r, _d, model, fps, _disp), rows in zip(jobs, parts):
            f.writelines(json.dumps(r) + "\n" for r in rows)
            train = [r["f1"] for r in rows if r["split"] == "train"]
            test = [r["f1"] for r in rows if r["split"] == "test"]
            print(f"{model:9} {fps:2d} FPS: {len(rows)} s, mean F1 train {sum(train) / len(train):.3f} "
                  f"test {sum(test) / len(test):.3f}", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
