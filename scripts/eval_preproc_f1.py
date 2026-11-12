"""Direction 2, step 4b (8/11): does the hardware resize change detection accuracy? F1 vs labels, same model, same frames.

  python -B scripts/eval_preproc_f1.py NPZ_DIR VIDEO_DIR LABEL_DIR MODEL.onnx

For every NPZ_DIR/<video>.npz (scripts/hw_dump_frames.py on the Edge): the same frame indices are decoded here with
OpenCV and resized with INTER_AREA (today's path). RT-DETRv2-S (full ONNX, CPU) runs on both 640x640 versions; F1 at IoU .5
against VN-Hard auto labels (LABEL_DIR/test_<video>.jsonl, DEIMv2-X, conf >= .5; car/motorcycle/bus/truck), predictions
score >= .6 (as scripts/cut_g3_eval.py), plus box agreement between the two paths.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detrac import match_frame  # noqa: E402

CLASSES, LABEL_CONF, THR = {2, 3, 5, 7}, .5, .6


def main():
    npz_dir, video_dir, label_dir, model = (Path(x) for x in sys.argv[1:5])
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    tot = {"opencv": [0, 0, 0], "hardware": [0, 0, 0]}
    agree = []
    for npz in sorted(npz_dir.glob("*.npz")):
        data = np.load(npz)
        name = npz.stem
        labels = {}
        for line in open(label_dir / f"test_{name}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            labels[r["frame"]] = r["boxes"]
        cap = cv2.VideoCapture(str(video_dir / f"{name}.mp4"))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sizes = np.array([[w, h]], np.int64)
        wanted, k, n = set(int(i) for i in data["index"]), 0, 0
        hw = dict(zip((int(i) for i in data["index"]), data["frames"]))
        while wanted:
            ok, bgr = cap.read()
            if not ok:
                break
            if k in wanted:
                wanted.discard(k)
                gt = [(x1 * w, y1 * h, x2 * w, y2 * h) for c, conf, x1, y1, x2, y2 in labels.get(k, [])
                      if conf >= LABEL_CONF and int(c) in CLASSES]
                rgb_ocv = cv2.cvtColor(cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                found = {}
                for path, rgb in (("opencv", rgb_ocv), ("hardware", hw[k])):
                    x = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]
                    sc, bx, lb = sess.run(["scores", "boxes", "labels"], {"images": x, "orig_target_sizes": sizes})
                    preds = [(float(c), tuple(map(float, b))) for c, b, l in zip(sc[0], bx[0], lb[0])
                             if c >= THR and int(l) in CLASSES]
                    found[path] = preds
                    tp, fp, fn = match_frame(gt, preds)
                    for j, v in enumerate((tp, fp, fn)):
                        tot[path][j] += v
                tp, fp, fn = match_frame([b for _, b in found["opencv"]], found["hardware"])
                agree.append(2 * tp / max(2 * tp + fp + fn, 1))
                n += 1
            k += 1
        print(f"{name}: {n} frames", flush=True)
    for path, (tp, fp, fn) in tot.items():
        print(f"{path:9s}: F1 {2 * tp / max(2 * tp + fp + fn, 1):.4f}  P {tp / max(tp + fp, 1):.4f}  R {tp / max(tp + fn, 1):.4f}"
              f"  (tp {tp} fp {fp} fn {fn})")
    print(f"box agreement hardware vs OpenCV: mean {np.mean(agree):.3f}, worst frame {min(agree):.3f} ({len(agree)} frames)")


if __name__ == "__main__":
    main()
