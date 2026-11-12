"""Run official candidate export with the matching architecture, then seal it.

Run in an isolated upstream export environment. Does not install/download code.
The emitted manifest is an admission contract, NOT a hardware/quality approval.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from candidate_artifact import seal


def command(kind, repo, checkpoint, output):
    if kind not in {"rtdetrv2_small", "lwdetr_large"}:
        raise ValueError("unsupported cloud model kind")
    if kind == "rtdetrv2_small":
        return [sys.executable, "tools/export_onnx.py", "-c", "configs/rtdetrv2/rtdetrv2_r18vd_sp2_120e_coco.yml",
                "-r", str(checkpoint), "-o", str(output), "-s", "640", "--check"], repo / "rtdetrv2_pytorch"
    # Architecture copied as arguments from official scripts/lwdetr_large_coco_export.sh.
    return [sys.executable, "main.py", "--batch_size", "1", "--encoder", "vit_small",
            "--vit_encoder_num_layers", "10", "--window_block_indexes", "0", "1", "3", "6", "7", "9",
            "--out_feature_indexes", "2", "4", "5", "9", "--dec_layers", "3", "--group_detr", "13",
            "--two_stage", "--projector_scale", "P3", "P5", "--hidden_dim", "384", "--sa_nheads", "12",
            "--ca_nheads", "24", "--dec_n_points", "4", "--bbox_reparam", "--lite_refpoint_refine",
            "--num_select", "300", "--dataset_file", "coco", "--square_resize_div_64", "--use_ema",
            "--eval", "--resume", str(checkpoint), "--output_dir", str(output.parent),
            "export_model"], repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["rtdetrv2_small", "lwdetr_large"])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo, checkpoint = args.repo.expanduser().resolve(strict=True), args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() or Path(str(output) + ".json").exists() or output.suffix != ".onnx":
        parser.error("choose a new .onnx output; overwrite is forbidden")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo):
        parser.error("upstream tracked checkout must be clean")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cloud-candidate-export-") as tmp:
        temporary = Path(tmp) / ("model.onnx" if args.kind == "rtdetrv2_small" else "inference_model.onnx")
        cmd, cwd = command(args.kind, repo, checkpoint, temporary)
        subprocess.run(cmd, cwd=cwd, check=True)
        # Exclusive create, including on a race; preserve failed export for diagnosis upstream.
        with temporary.open("rb") as source, output.open("xb") as target:
            shutil.copyfileobj(source, target)
    seal(output, args.kind, repo=repo, checkpoint=checkpoint, export_evidence={"argv": cmd, "hardware_gate": "PENDING"})


if __name__ == "__main__":
    main()
