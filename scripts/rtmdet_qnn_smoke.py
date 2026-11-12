"""Real-frame HTP diagnostic only. No CPU fallback, FPS claim or pipeline admission."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.rtmdet_qnn_diagnostic import (BUNDLE_HASHES, ENCODINGS, MODEL_ID,
    PREPROCESSING_PROFILES, REFERENCE_CONFIG, REFERENCE_HEAD, preprocessing_contract,
    verify_bundle, prepare_frame, decode_heads, run_native, dequantize)


def validate_args(args):
    if not args.allow_unverified_contract:
        raise ValueError("requires --allow-unverified-contract; preprocessing/head semantics not yet verified")
    if not 1 <= args.repeats <= 5 or args.frame_index < 0 or not 0 < args.confidence <= 1:
        raise ValueError("repeats1..5, nonnegative frame index and confidence>0..1 required")
    if not 0 < args.timeout_s <= 300:
        raise ValueError("diagnostic timeout must be >0..300 seconds")
    preprocessing_contract(getattr(args, "preprocessing", PREPROCESSING_PROFILES[0]))


def load_frame(args):
    import cv2
    capture = cv2.VideoCapture(str(args.video))
    try:
        if args.frame_index and not capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index):
            raise ValueError("cannot seek representative frame")
        ok, image = capture.read()
        if not ok or image is None or image.shape[0] == image.shape[1]:
            raise ValueError("cannot read non-square representative video frame")
    finally:
        capture.release()
    return image


def smoke(args, *, image=None, work=None):
    validate_args(args)
    folder = verify_bundle(args.artifact_dir)
    if image is None:
        image = load_frame(args)
    profile = getattr(args, "preprocessing", PREPROCESSING_PROFILES[0])
    prepared = prepare_frame(image, allow_unverified_contract=True, preprocessing=profile)
    if work is None:
        parent = Path(args.output_parent).expanduser().resolve(strict=True)
        work = Path(tempfile.mkdtemp(prefix="rtmdet_qnn_real_", dir=parent))
    else:
        work = Path(work).resolve(strict=True)
    print(f"Smoke directory: {work}", flush=True)
    outputs, runtime = run_native(folder, args.runtime_root, work, prepared.native,
                                 args.repeats, args.timeout_s)
    boxes = decode_heads(outputs[-1], prepared, confidence=args.confidence,
                         allow_unverified_contract=True)
    from PIL import Image, ImageDraw
    canvas = Image.fromarray(image[..., ::-1])
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        bounds = (box.left * prepared.width, box.top * prepared.height,
                  box.right * prepared.width, box.bottom * prepared.height)
        draw.rectangle(bounds, outline=(0, 255, 0), width=2)
        draw.text(bounds[:2], f"{box.class_name} {box.confidence:.2f}", fill=(255, 255, 0))
    with (work / "boxes.png").open("xb") as target:
        canvas.save(target, format="PNG")
    stats = {}
    for name, value in outputs[-1].items():
        floating = dequantize(value, name)
        stats[name] = {"shape": list(value.shape), "native_dtype": str(value.dtype),
                       "native_min": int(value.min()), "native_max": int(value.max()),
                       "dequant_min": float(floating.min()), "dequant_max": float(floating.max()),
                       "scale": ENCODINGS[name][1], "zero_point": ENCODINGS[name][2]}
    evidence = {"model_id": MODEL_ID, "bundle_hashes": BUNDLE_HASHES,
        "runtime": "qnn_htp_native_diagnostic", "runtime_gate": "NATIVE_OUTPUTS_VALID",
        "contract_gate": "UNVERIFIED_PREPROCESS_AND_HEAD_SEMANTICS",
        "quality_status": "NOT_EVALUATED", "parity": "PENDING", "pipeline_admitted": False,
        "benchmark": False, "fps": None, "tracking": False, "repeats_same_frame": args.repeats,
        "source": {"video": str(Path(args.video).resolve()), "frame_index": args.frame_index,
                   "shape": list(image.shape), "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()},
        "runtime_evidence": runtime, "preprocessing_profile": profile,
        "native_input_sha256": hashlib.sha256(prepared.native.tobytes()).hexdigest(),
        "input_clipped_fraction": prepared.clipped_fraction,
        "experimental_contract": {**preprocessing_contract(profile),
            "native_layout": "NCHW UINT8", "classification": "sigmoid(logits)",
            "regression": "already stride-scaled LTRB distances, prior offset0, strides8/16/32",
            "label_map": "assumed_coco80_contiguous", "confidence": args.confidence,
            "nms_iou": .65, "top_k": 300, "reference_config": REFERENCE_CONFIG,
            "reference_head": REFERENCE_HEAD, "artifact_source_match": "UNVERIFIED"},
        "tensor_stats": stats, "last_boxes": [asdict(box) for box in boxes],
        "overlay": str(work / "boxes.png")}
    with (work / "report.json").open("x", encoding="utf-8") as target:
        json.dump(evidence, target, indent=2, allow_nan=False)
    print(json.dumps({"report": str(work / "report.json"), "runtime_gate": evidence["runtime_gate"],
                      "contract_gate": evidence["contract_gate"], "detections": len(boxes),
                      "preprocessing_profile": profile,
                      "input_clipped_fraction": prepared.clipped_fraction}))
    return 0


def compare_preprocessing(args):
    validate_args(args)
    verify_bundle(args.artifact_dir)
    image = load_frame(args)  # decode the video exactly once for both cases
    parent = Path(args.output_parent).expanduser().resolve(strict=True)
    work = Path(tempfile.mkdtemp(prefix="rtmdet_qnn_ab_", dir=parent))
    print(f"A/B directory: {work}", flush=True)
    cases = []
    for profile in PREPROCESSING_PROFILES:
        case_dir = work / profile
        case_dir.mkdir()
        case_args = argparse.Namespace(**vars(args))
        case_args.preprocessing = profile
        smoke(case_args, image=image, work=case_dir)
        report = json.loads((case_dir / "report.json").read_text(encoding="utf-8"))
        cases.append(report)
    # Fail closed if the frame, artifact, runtime or decoder controls differ.
    for key in ("source", "bundle_hashes", "model_id", "repeats_same_frame"):
        if cases[0][key] != cases[1][key]:
            raise ValueError(f"A/B control mismatch: {key}")
    # --version text includes a process ID; compare stable SDK identity/hashes,
    # not the full text of two separate invocations.
    for key in ("sdk_version", "runner_sha256", "htp_sha256"):
        if cases[0]["runtime_evidence"][key] != cases[1]["runtime_evidence"][key]:
            raise ValueError(f"A/B runtime control mismatch: {key}")
    for key in ("resize", "classification", "regression", "label_map", "confidence", "nms_iou", "top_k"):
        if cases[0]["experimental_contract"][key] != cases[1]["experimental_contract"][key]:
            raise ValueError(f"A/B decoder/geometry control mismatch: {key}")
    summary = {"experiment": "same_frame_preprocessing_ab", "source": cases[0]["source"],
               "bundle_hashes": BUNDLE_HASHES, "runtime_gate": "NATIVE_OUTPUTS_VALID_BOTH_CASES",
               "contract_gate": "UNVERIFIED_PREPROCESS_AND_HEAD_SEMANTICS", "winner": None,
               "quality_status": "NOT_EVALUATED", "pipeline_admitted": False, "benchmark": False,
               "fps": None, "controls_verified": True, "cases": []}
    for report in cases:
        boxes = report["last_boxes"]
        summary["cases"].append({"preprocessing_profile": report["preprocessing_profile"],
            "report": str(work / report["preprocessing_profile"] / "report.json"), "overlay": report["overlay"],
            "detections": len(boxes), "classes": dict(Counter(b["class_name"] for b in boxes)),
            "boxes_over_80percent_image": sum((b["right"]-b["left"])*(b["bottom"]-b["top"]) > .8 for b in boxes),
            "input_clipped_fraction": report["input_clipped_fraction"],
            "native_input_sha256": report["native_input_sha256"]})
    with (work / "ab_report.json").open("x", encoding="utf-8") as target:
        json.dump(summary, target, indent=2, allow_nan=False)
    print(json.dumps({"ab_report": str(work / "ab_report.json"), "cases": summary["cases"], "winner": None}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--confidence", type=float, default=.10)
    parser.add_argument("--preprocessing", choices=PREPROCESSING_PROFILES, default=PREPROCESSING_PROFILES[0])
    parser.add_argument("--compare-preprocessing", action="store_true")
    parser.add_argument("--allow-unverified-contract", action="store_true")
    args = parser.parse_args()
    return compare_preprocessing(args) if args.compare_preprocessing else smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
