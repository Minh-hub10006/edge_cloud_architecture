"""Direction 2 step 1 (8/11): what the hardware 640x640 (stretched, qtivtransform gles) frames do to mode A and to DRL state.

  python -B scripts/eval_preproc_modeA_state.py NPZ_DIR VIDEO_DIR LABEL_DIR YOLO11S320.onnx

Per frame index in NPZ_DIR/<video>.npz (scripts/hw_dump_frames.py), the full-size OpenCV frame (today) vs the hardware frame:
- mode A: YOLO11s320 float ONNX (the DLC's source, sha 62ff9e6d) through yolo11s_contract.prepare/decode, which
  letterboxes whatever it gets; F1 at IoU .5 vs VN auto labels (vehicles), at confidence .25 and .5.
- state: scene_features (brightness, contrast, laplacian_small, motion vs the previous saved frame, 200 ms apart at
  every=3 / 15 fps) and laplacian_variance at blur_width 320, as edge_agent/control_v1.py computes them.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import yolo11s_contract as yc  # noqa: E402
from detrac import match_frame  # noqa: E402
from edge_agent.resources_v2 import scene_features  # noqa: E402
from edge_agent.state_metrics import laplacian_variance  # noqa: E402

CLASSES, LABEL_CONF, THRS = {2, 3, 5, 7}, .5, (.25, .5)


def state(bgr, prev_gray):
    feats, gray = scene_features(bgr, prev_gray)
    h, w = bgr.shape[:2]
    feats["laplacian_variance"] = laplacian_variance(cv2.resize(bgr, (320, max(1, round(h * 320 / w)))) if w != 320 else bgr)
    return feats, gray


def main():
    npz_dir, video_dir, label_dir, model = (Path(x) for x in sys.argv[1:5])
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    PATHS = ("opencv", "hardware", "unstretched")
    tot = {(p, t): [0, 0, 0] for p in PATHS for t in THRS}
    feats = {p: {} for p in PATHS}
    for npz in sorted(npz_dir.glob("*.npz")):
        data, name = np.load(npz), npz.stem
        labels = {}
        for line in open(label_dir / f"test_{name}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            labels[r["frame"]] = r["boxes"]
        hw = dict(zip((int(i) for i in data["index"]), data["frames"]))
        cap = cv2.VideoCapture(str(video_dir / f"{name}.mp4"))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        k, prev = 0, dict.fromkeys(PATHS)
        # "unstretched": the 640x640 frame squeezed back to the camera's aspect (640 on the long side), cheap on CPU
        back = (round(640 * w / max(w, h)), round(640 * h / max(w, h)))
        while k <= max(hw):
            ok, full = cap.read()
            if not ok:
                break
            if k in hw:
                gt = [(x1 * w, y1 * h, x2 * w, y2 * h) for c, conf, x1, y1, x2, y2 in labels.get(k, [])
                      if conf >= LABEL_CONF and int(c) in CLASSES]
                small = np.ascontiguousarray(hw[k][:, :, ::-1])
                for path, bgr in (("opencv", full), ("hardware", small),
                                  ("unstretched", cv2.resize(small, back, interpolation=cv2.INTER_AREA))):
                    prep = yc.prepare(bgr)
                    out = dict(zip(names, sess.run(names, {sess.get_inputs()[0].name: prep.tensor})))
                    dets = yc.decode({n: out[n] for n in ("boxes", "scores")}, prep, confidence=min(THRS))
                    for t in THRS:
                        preds = [(d.confidence, (d.left * w, d.top * h, d.right * w, d.bottom * h)) for d in dets
                                 if d.confidence >= t and d.class_id in CLASSES]
                        for j, v in enumerate(match_frame(gt, preds)):
                            tot[(path, t)][j] += v
                    f, prev[path] = state(bgr, prev[path])
                    for key, v in f.items():
                        feats[path].setdefault(key, []).append(v)
            k += 1
        print(f"{name}: {len(hw)} frames", flush=True)
    print("mode A (YOLO11s320, letterbox of whatever frame it gets):")
    for t in THRS:
        for path in PATHS:
            tp, fp, fn = tot[(path, t)]
            print(f"  conf >= {t:.2f} {path:9s}: F1 {2 * tp / max(2 * tp + fp + fn, 1):.4f}  P {tp / max(tp + fp, 1):.4f}"
                  f"  R {tp / max(tp + fn, 1):.4f}  (tp {tp} fp {fp} fn {fn})")
    print("state features (full OpenCV frame vs hardware 640x640):")
    for other in PATHS[1:]:
        for key in feats["opencv"]:
            a, b = np.array(feats["opencv"][key]), np.array(feats[other][key])
            corr = np.corrcoef(a, b)[0, 1] if a.std() and b.std() else float("nan")
            print(f"  {other:11s} {key:18s}: mean {a.mean():9.2f} vs {b.mean():9.2f} (ratio {b.mean() / a.mean():.2f}), "
                  f"correlation {corr:.3f}, {len(a)} samples")


if __name__ == "__main__":
    main()
