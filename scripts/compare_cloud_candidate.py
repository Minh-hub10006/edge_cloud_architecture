"""Compare official PyTorch deploy output with a sealed ONNX on one real image.

Run in the matching upstream export environment. No random-weight reference.
This is numerical export parity, not task accuracy/mAP or tracking quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

from candidate_artifact import MODEL_IDS, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["rtdetrv2_small", "lwdetr_large"])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cuda", action="store_true", help="also requires CUDA ONNX Runtime")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import onnxruntime as ort
    from inference.cloud_detr import CloudDetrAdapter
    from inference.model_artifact import load_artifact, OUTPUT_NAMES
    artifact, manifest = load_artifact(str(args.artifact), args.kind)
    if sha256(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("reference checkpoint differs from artifact provenance")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if revision != manifest["source_revision"]:
        raise ValueError("reference checkout differs from artifact provenance")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.repo, text=True).strip():
        raise ValueError("reference checkout has tracked changes; use the sealed clean revision")
    pixels = cv2.imread(str(args.image))
    if pixels is None or pixels.shape[0] == pixels.shape[1]:
        raise ValueError("provide non-square real image")
    adapter = object.__new__(CloudDetrAdapter)
    adapter.kind = args.kind
    feed = adapter.prepare(pixels)
    # Official checkpoints must be trusted; PyTorch checkpoint format may use pickle.
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if args.kind == "rtdetrv2_small":
        sys.path.insert(0, str(args.repo / "rtdetrv2_pytorch"))
        from src.core import YAMLConfig
        config = YAMLConfig(str(args.repo / manifest["source_config"]))
        model = config.model
        model.load_state_dict(checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"], strict=True)
        model, post = model.deploy().eval(), config.postprocessor.deploy().eval()
        with torch.no_grad():
            reference = post(model(torch.from_numpy(feed["images"])), torch.from_numpy(feed["orig_target_sizes"]))
    else:
        sys.path.insert(0, str(args.repo))
        from main import get_args_parser
        from models import build_model
        from export_cloud_candidate import command
        argv, _ = command(args.kind, args.repo, args.checkpoint, Path("unused/inference_model.onnx"))
        config = get_args_parser().parse_args(argv[2:])
        config.device = "cpu"
        model, _, _ = build_model(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().export()
        with torch.no_grad():
            reference = model(torch.from_numpy(feed["input"]))
    providers = ["CPUExecutionProvider"]
    if args.cuda:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider unavailable")
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(artifact), providers=providers)
    session.disable_fallback()
    if args.cuda and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA initialization failed")
    actual = session.run(list(OUTPUT_NAMES[args.kind]), feed)
    if len(reference) != len(actual):
        raise ValueError("reference output count differs from ONNX")
    errors = {}
    failed = False
    for name, wanted, found in zip(OUTPUT_NAMES[args.kind], reference, actual):
        wanted = wanted.detach().cpu().numpy()
        tolerance = .1 if name == "boxes" else 1e-3
        try:
            if args.kind == "rtdetrv2_small" and name == "labels":
                np.testing.assert_array_equal(found, wanted)
            else:
                np.testing.assert_allclose(found, wanted, rtol=1e-3, atol=tolerance)
            errors[name] = {"status": "PASS", "max_abs": float(np.max(np.abs(found - wanted)))}
        except AssertionError as exc:
            failed = True
            errors[name] = {"status": "FAIL", "error": str(exc)}
    report = {"model_id": MODEL_IDS[args.kind], "artifact_sha256": sha256(artifact),
              "image_sha256": sha256(args.image), "providers": session.get_providers(),
              "parity": "FAIL" if failed else "PASS", "outputs": errors,
              "quality_status": "NOT_EVALUATED", "scope": "one image raw export parity; not accuracy/throughput"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
    print(json.dumps({"parity": report["parity"], "report": str(args.report)}))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
