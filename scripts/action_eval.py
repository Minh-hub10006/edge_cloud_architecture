"""UC4 action model choice (P3, 11/11): ST-GCN++ vs PoseC3D (MMAction2, Apache-2.0, NTU60 2D keypoints) on skeletons
from RTMO-s - the chain the Cloud will run - on public surveillance-like clips (academic use, measured on the laptop only).

  python -B scripts/action_eval.py pose RTMO.onnx VIDEO_DIR OUT.pkl [--fps 10]
  python -B scripts/action_eval.py onnx POSES.pkl OUT.onnx --config C --checkpoint W   # ST-GCN++ for Triton
  python -B scripts/action_eval.py score POSES.pkl --task fall|fight --model stgcnpp|posec3d --config C --checkpoint W
                                  [--fps 10]

pose:  RTMO-s (scripts/pose_eval.PoseModel, conf .3) on every video in VIDEO_DIR sampled at --fps -> per frame the
       people's 17 COCO keypoints and scores (cached: the slow part, done once for both action models).
score: one clip = one video (UR Fall 2-10 s, fights 2 s), re-sampled to --fps (5 = what an Edge trigger would send),
       1 test clip (production; the published 89-94 % use 10), persons ordered by score (no tracker here).
       Score of a clip: fall = P(NTU60 "falling"); fight = P(punching) + P(kicking) + P(pushing).
       Label: fall - file name starts with "fall" (UR Fall: fall-NN / adl-NN); fight - a parent folder named "fight"
       (case-insensitive; "noFight" is negative). -> AUC, and recall / false alarms at the threshold with <= 10 %
       false alarms, at top-1 == the class, ms per clip on CPU.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CLASSES = {"fall": [42], "fight": [49, 50, 51]}   # NTU60 (0-based): falling; punching/slapping, kicking, pushing


def extract(onnx, video_dir, out, fps):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pose_eval import PoseModel
    net, poses = PoseModel(onnx, conf=.3), {}
    videos = sorted(p for p in Path(video_dir).rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mpeg", ".mov"))
    for k, v in enumerate(videos):
        cap = cv2.VideoCapture(str(v))
        src = cap.get(cv2.CAP_PROP_FPS) or 30
        step, frames, i, shape = max(1, round(src / fps)), [], 0, None
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                shape = bgr.shape[:2]
                people = net(bgr)
                frames.append((np.array([p[2][:, :2] for p in people], np.float32).reshape(-1, 17, 2),
                               np.array([p[2][:, 2] for p in people], np.float32).reshape(-1, 17)))
            i += 1
        poses[str(v.relative_to(video_dir))] = {"img_shape": shape, "fps": src / step, "frames": frames}
        print(f"  {k + 1}/{len(videos)} {v.name}: {len(frames)} frames", flush=True)
    Path(out).write_bytes(pickle.dumps(poses))


def label(name, task):
    parts = [p.lower() for p in Path(name).parts]
    return parts[-1].startswith("fall") if task == "fall" else "fight" in parts[:-1]


def auc(pos, neg):
    return float(np.mean([(p > n) + .5 * (p == n) for p in pos for n in neg])) if pos and neg else float("nan")


def score(poses_pkl, task, config, checkpoint, fps):
    from mmaction.apis import inference_skeleton, init_recognizer
    from mmengine import Config
    cfg = Config.fromfile(config)
    for t in cfg.test_pipeline:
        if t["type"] == "UniformSampleFrames":
            t["num_clips"] = 1
    model = init_recognizer(cfg, checkpoint, device="cpu")
    poses, rows = pickle.loads(Path(poses_pkl).read_bytes()), []
    for name, clip in poses.items():
        step = max(1, round(clip["fps"] / fps))
        frames = [{"keypoints": kp, "keypoint_scores": sc} for kp, sc in clip["frames"][::step]]
        if not any(len(f["keypoints"]) for f in frames):
            rows.append((name, label(name, task), 0.0, -1, 0.0))   # nobody seen: the chain cannot fire
            continue
        t = time.perf_counter()
        out = inference_skeleton(model, frames, clip["img_shape"])
        ms = (time.perf_counter() - t) * 1000
        p = out.pred_score.numpy()
        p = p if abs(p.sum() - 1) < 1e-3 else np.exp(p - p.max()) / np.exp(p - p.max()).sum()
        rows.append((name, label(name, task), float(p[CLASSES[task]].sum()), int(p.argmax()), ms))
    pos = [s for _, y, s, _, _ in rows if y]
    neg = [s for _, y, s, _, _ in rows if not y]
    thr = float(np.quantile(neg, .9)) if neg else .5            # <= 10 % false alarms on the negatives
    top1 = lambda y: sum(top in CLASSES[task] for _, yy, _, top, _ in rows if yy == y)
    print(f"{Path(checkpoint).name[:40]} task {task} @ {fps} fps: {len(pos)} positive, {len(neg)} negative clips")
    print(f"  AUC {auc(pos, neg):.3f}")
    print(f"  threshold {thr:.3f} (10 % false alarms): caught {sum(s > thr for s in pos)}/{len(pos)}, "
          f"false alarms {sum(s > thr for s in neg)}/{len(neg)}")
    print(f"  top-1 is the class: caught {top1(True)}/{len(pos)}, false alarms {top1(False)}/{len(neg)}")
    print(f"  CPU {np.median([r[4] for r in rows if r[4]]):.0f} ms per clip (laptop)")
    return rows


def export(poses_pkl, out, config, checkpoint, fps=5):
    """ST-GCN++ -> ONNX (input (B, 2, 100, 17, 3), output softmax (B, 60)), then parity on the cached clips:
    the ONNX vs PyTorch on inference.action_skeleton.prepare_clip inputs."""
    import onnxruntime as ort
    import torch
    from mmaction.apis import init_recognizer
    from mmengine import Config
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference.action_skeleton import prepare_clip
    cfg = Config.fromfile(config)
    for t in cfg.test_pipeline:
        if t["type"] == "UniformSampleFrames":
            t["num_clips"] = 1   # as production (and `score`)
    model = init_recognizer(cfg, checkpoint, device="cpu").eval()

    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(model.cls_head(model.backbone(x)), -1)
    model.requires_grad_(False)   # the adaptive graph is used as a constant: tracing refuses it with grad on
    with torch.no_grad():
        torch.onnx.export(Net(), torch.zeros(1, 2, 100, 17, 3), out, input_names=["skeleton"], output_names=["prob"],
                          dynamic_axes={"skeleton": {0: "batch"}, "prob": {0: "batch"}}, opset_version=17, dynamo=False)
    sess, worst = ort.InferenceSession(out, providers=["CPUExecutionProvider"]), 0.0
    for name, clip in pickle.loads(Path(poses_pkl).read_bytes()).items():
        step = max(1, round(clip["fps"] / fps))
        frames = clip["frames"][::step]
        if not any(len(kp) for kp, _ in frames):
            continue
        x = prepare_clip([np.concatenate([kp, sc[..., None]], -1) for kp, sc in frames], clip["img_shape"])
        with torch.no_grad():
            ref = Net()(torch.from_numpy(x)).numpy()[0]
        worst = max(worst, float(np.abs(sess.run(None, {"skeleton": x})[0][0] - ref).max()))
    # 11/11: prepare_clip = MMAction2's test pipeline input within 0.001 (it rounds keypoints to float16, which moves
    # its probabilities by up to 0.015); here the ONNX vs PyTorch on the same input
    print(f"{out}: ONNX vs PyTorch worst |prob diff| {worst:.4f} ({'PASS' if worst < .01 else 'CHECK'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pose", "score", "onnx"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--task", choices=list(CLASSES))
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    a = ap.parse_args()
    if a.cmd == "onnx":
        export(a.args[0], a.args[1], a.config, a.checkpoint)
    elif a.cmd == "pose":
        extract(a.args[0], a.args[1], a.args[2], a.fps)
    else:
        score(a.args[0], a.task, a.config, a.checkpoint, a.fps)


if __name__ == "__main__":
    sys.exit(main())
