"""Known-source RTMDet-S export; independent MMDetection oracle, no NPU admission.

Writes only a new output directory. Does not change adapters, SDKs or catalogs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.candidate_artifact import seal, sha256

REVISION = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
CHECKPOINT_SHA256 = "387a891e157cf0ab57d76b3ffc17bf77247089d672532427930b3140f9e789d6"
CONFIG = "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"


def load_strict_checkpoint(model, state):
    """Old checkpoints persist mean/std; new MMEngine buffers do not.

    Verify these exact constants, then omit only nonpersistent duplicates.
    Every learned parameter still goes through strict=True.
    """
    import torch
    state = dict(state)
    expected = model.state_dict()
    buffers = dict(model.named_buffers())
    omitted = []
    for name in ("data_preprocessor.mean", "data_preprocessor.std"):
        if name not in state or name not in buffers:
            raise ValueError(f"missing official preprocessing constant: {name}")
        torch.testing.assert_close(state[name], buffers[name], rtol=0, atol=0)
        if name not in expected:
            state.pop(name)
            omitted.append(name)
    model.load_state_dict(state, strict=True)
    return omitted


def compare_detections(expected, actual, *, coordinate_atol=2e-4, score_atol=1e-4):
    """One-to-one matching, including class and score; empty is not evidence."""
    import numpy as np
    if not expected or len(expected) != len(actual):
        raise ValueError(f"detection counts differ or empty oracle: {len(expected)} vs {len(actual)}")
    unused = set(range(len(actual)))
    worst_coordinate = worst_score = 0.0
    for box in expected:
        candidates = []
        wanted = np.array([box.left, box.top, box.right, box.bottom])
        for index in unused:
            found = actual[index]
            if found.class_id != box.class_id:
                continue
            coords = np.array([found.left, found.top, found.right, found.bottom])
            delta = float(np.max(np.abs(coords - wanted)))
            score_delta = abs(found.confidence - box.confidence)
            if np.isfinite(delta) and np.isfinite(score_delta):
                candidates.append((max(delta / coordinate_atol, score_delta / score_atol), index, delta, score_delta))
        if not candidates:
            raise ValueError(f"no matching class {box.class_id}")
        distance, index, delta, score_delta = min(candidates)
        if distance > 1:
            raise ValueError(f"box/class/score parity failed: class={box.class_id}, xyxy={delta}, score={score_delta}")
        unused.remove(index)
        worst_coordinate = max(worst_coordinate, delta)
        worst_score = max(worst_score, score_delta)
    return {"gate": "PASS", "detections": len(expected),
            "max_normalized_coordinate_error": worst_coordinate,
            "max_score_error": worst_score, "coordinate_atol": coordinate_atol,
            "score_atol": score_atol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", type=Path)
    media.add_argument("--video", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=.1)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16 or args.frame_index < 0 or not 0 < args.confidence < 1:
        parser.error("invalid threads/frame-index/confidence")
    repo = args.repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    if revision != REVISION or dirty or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("source revision/clean checkout/official checkpoint SHA gate failed")
    if output.exists():
        raise FileExistsError("output directory must be new; no overwrites")
    sys.path.insert(0, str(repo))
    import copy
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from mmengine.dataset import Compose
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import get_test_pipeline_cfg
    from inference.detection_decode import COCO_NAMES, normalize, rtmdet_decode
    from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    media_path = (args.image or args.video).resolve(strict=True)
    if args.video:
        capture = cv2.VideoCapture(str(media_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
            ok, image = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError("cannot decode requested video frame")
    else:
        image = cv2.imread(str(media_path))
    if image is None or image.shape[0] == image.shape[1]:
        raise ValueError("representative non-square image required")

    model = init_detector(str(repo / CONFIG), str(checkpoint), device="cpu", palette="coco").eval()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["state_dict"]
    verified_nonpersistent = load_strict_checkpoint(model, state)
    if tuple(model.dataset_meta["classes"]) != COCO_NAMES or model.bbox_head.num_classes != 80:
        raise ValueError("official checkpoint COCO80 class contract mismatch")
    model.bbox_head.test_cfg.score_thr = args.confidence
    model.bbox_head.test_cfg.max_per_img = 300

    # Oracle uses upstream's complete image pipeline and decoder, not our export.
    pipeline_cfg = copy.deepcopy(get_test_pipeline_cfg(model.cfg))
    pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
    data = Compose(pipeline_cfg)({"img": image, "img_id": 0})
    official_batch = model.data_preprocessor(
        {"inputs": [data["inputs"]], "data_samples": [data["data_samples"]]}, training=False)
    adapter = object.__new__(RtmdetSmallSnpeAdapter)
    adapter.manifest = {"input_layout": "nchw"}
    prepared = adapter.prepare_frame(image)
    np.testing.assert_allclose(prepared.tensor, official_batch["inputs"].numpy(), atol=1e-6, rtol=1e-6)
    meta = data["data_samples"].metainfo
    np.testing.assert_allclose([prepared.scale_x, prepared.scale_y], meta["scale_factor"], atol=1e-7)
    with torch.no_grad():
        official = inference_detector(model, image).pred_instances
    reference_boxes = normalize(official.bboxes.numpy(), official.scores.numpy(), official.labels.numpy(),
                                width=prepared.width, height=prepared.height)
    if not reference_boxes:
        raise ValueError("oracle returned no boxes; choose a representative image")

    dummy = torch.from_numpy(prepared.tensor)
    with torch.no_grad():
        cls, _ = model.bbox_head(model.extract_feat(dummy))
        priors = model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls], device="cpu")

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = model
            for index, prior in enumerate(priors):
                self.register_buffer(f"prior_{index}", prior[:, :2][None])

        def forward(self, value):
            cls, reg = self.detector.bbox_head(self.detector.extract_feat(value))
            boxes, scores = [], []
            for index, (c, d) in enumerate(zip(cls, reg)):
                distance = d.permute(0, 2, 3, 1).reshape(1, -1, 4)
                prior = getattr(self, f"prior_{index}")
                boxes.append(torch.cat((prior - distance[..., :2], prior + distance[..., 2:]), -1))
                scores.append(c.permute(0, 2, 3, 1).reshape(1, -1, 80).sigmoid())
            return torch.cat(boxes, 1), torch.cat(scores, 1)

    # Border clipping is image-dependent and belongs before NMS on the host.
    def decode(values):
        height, width = meta["img_shape"][:2]
        clipped = values[0].copy()
        clipped[..., [0, 2]] = np.clip(clipped[..., [0, 2]], 0, width)
        clipped[..., [1, 3]] = np.clip(clipped[..., [1, 3]], 0, height)
        return rtmdet_decode({"boxes": clipped, "scores": values[1]}, width=prepared.width,
                             height=prepared.height, scale_x=prepared.scale_x, scale_y=prepared.scale_y,
                             confidence=args.confidence, top_k=300)

    exporter = Export().eval()
    with torch.no_grad():
        raw_reference = tuple(x.numpy() for x in exporter(dummy))
    torch_gate = compare_detections(reference_boxes, decode(raw_reference))
    output.mkdir(parents=True)
    artifact = output / "rtmdet_small_float.onnx"
    with torch.no_grad():
        torch.onnx.export(exporter, dummy, str(artifact), input_names=["input"], output_names=["boxes", "scores"],
                          opset_version=13, do_constant_folding=True)
    onnx.checker.check_model(str(artifact))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(["boxes", "scores"], {"input": prepared.tensor})
    if [list(x.shape) for x in actual] != [[1, 8400, 4], [1, 8400, 80]]:
        raise ValueError("ONNX IO shape gate failed")
    # Raw boxes are pixels: use an absolute subpixel bound, not relative
    # error around zero (negative/border coords are allowed before clipping).
    np.testing.assert_allclose(actual[0], raw_reference[0], rtol=0, atol=.1)
    np.testing.assert_allclose(actual[1], raw_reference[1], rtol=0, atol=1e-4)
    raw_errors = {"boxes_max_absolute_pixels": float(np.max(np.abs(actual[0] - raw_reference[0]))),
                  "scores_max_absolute": float(np.max(np.abs(actual[1] - raw_reference[1]))),
                  "boxes_atol_pixels": .1, "scores_atol": 1e-4}
    decoded = decode(actual)
    onnx_gate = compare_detections(reference_boxes, decoded)

    def overlay(name, boxes):
        canvas = image.copy()
        for box in boxes:
            start = (round(box.left * prepared.width), round(box.top * prepared.height))
            stop = (round(box.right * prepared.width), round(box.bottom * prepared.height))
            cv2.rectangle(canvas, start, stop, (0, 255, 0), 2)
            cv2.putText(canvas, f"{box.class_name} {box.confidence:.2f}", start,
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        if not cv2.imwrite(str(output / name), canvas):
            raise OSError("overlay write failed")

    if not cv2.imwrite(str(output / "source.png"), image):
        raise OSError("source frame write failed")
    overlay("pytorch_reference.png", reference_boxes)
    overlay("onnx_reference.png", decoded)
    np.save(output / "reference_input.npy", prepared.tensor, allow_pickle=False)
    np.savez(output / "reference_outputs.npz", boxes=actual[0], scores=actual[1])
    evidence = {"preprocess_vs_mmdetection": "PASS", "strict_checkpoint_load": "PASS",
                "verified_nonpersistent_checkpoint_buffers": verified_nonpersistent,
                "torch_decoder_vs_mmdetection": torch_gate, "onnx_decoder_vs_mmdetection": onnx_gate,
                "torch_onnx_raw_parity": "PASS", "confidence": args.confidence,
                "raw_parity_errors": raw_errors,
                "nms_iou": .65, "top_k": 300, "image_sha256": sha256(output / "source.png"),
                "host_decoder_required": "clip xyxy to per-frame upstream img_shape BEFORE class-wise NMS",
                "preprocessing": "BGR keep-ratio resize 640, right/bottom pad114, normalize mean/std, NCHW float32",
                "input_mean": [103.53, 116.28, 123.675], "input_std": [57.375, 57.12, 58.395]}
    seal(artifact, "rtmdet_small", repo=repo, checkpoint=checkpoint, export_evidence=evidence)
    report = {"schema_version": 1, "artifact": artifact.name, "artifact_sha256": sha256(artifact),
              "source_revision": revision, "checkpoint_sha256": CHECKPOINT_SHA256,
              "source_media_sha256": sha256(media_path), "frame_index": args.frame_index if args.video else None,
              "image_shape": list(image.shape), "oracle_img_shape": list(meta["img_shape"]),
              "export_gate": "PASS", "evidence": evidence, "boxes": [asdict(b) for b in decoded],
              "hardware_gate": "NOT_RUN", "quantization_gate": "NOT_RUN", "quality_status": "NOT_EVALUATED",
              "pipeline_gate": "NOT_RUN", "production_ready": False}
    with (output / "rebuild_report.json").open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
    with (output / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as target:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                target.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({"output": str(output), "export_gate": "PASS", "detections": len(decoded),
                      "hardware_gate": "NOT_RUN", "quality_status": "NOT_EVALUATED"}))


if __name__ == "__main__":
    main()
