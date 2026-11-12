"""Seal/probe real candidate exports. CPU sealing is NOT a CUDA capacity test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.model_artifact import MODEL_IDS, OUTPUT_NAMES, sha256, candidate_model_id, _check_external_references

SOURCES = {
    "rtdetrv2_small": ("https://github.com/lyuwenyu/RT-DETR", "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp2_120e_coco.yml"),
    "lwdetr_large": ("https://github.com/Atten4Vis/LW-DETR", "scripts/lwdetr_large_coco_export.sh"),
    "dfine_x": ("https://github.com/Peterande/D-FINE", "configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml"),
    "rtmdet_small": ("https://github.com/open-mmlab/mmdetection", "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"),
}


def probe_onnx(path, kind):
    import numpy as np
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    names = {item.name for item in session.get_inputs()}
    rtdetr_io = kind in {"rtdetrv2_small", "dfine_x"}
    image_name = "images" if rtdetr_io else "input"
    feed = {image_name: np.zeros((1, 3, 640, 640), dtype=np.float32)}
    if rtdetr_io:
        feed["orig_target_sizes"] = np.array([[1280, 720]], dtype=np.int64)
    if names != set(feed) or {item.name for item in session.get_outputs()} != set(OUTPUT_NAMES[kind]):
        raise ValueError("unexpected ONNX input/output names")
    outputs = session.run(list(OUTPUT_NAMES[kind]), feed)
    if any(not np.isfinite(value).all() for value in outputs):
        raise ValueError("nonfinite ONNX outputs")
    if kind == "lwdetr_large":
        from inference.detection_decode import lwdetr_decode
        lwdetr_decode(dict(zip(OUTPUT_NAMES[kind], outputs)), .1, 300)
    if kind == "dfine_x":
        from inference.detection_decode import rtdetr_decode
        rtdetr_decode(dict(zip(OUTPUT_NAMES[kind], outputs)), 1280, 720, .1, 300)
    return {name: list(value.shape) for name, value in zip(OUTPUT_NAMES[kind], outputs)}


IMPORT_CONTRACTS = {
    "lwdetr_large": {
        "inputs": {"input": {"dtype": "FLOAT", "shape": [1, 3, 640, 640]}},
        "output_dtypes": {"dets": "FLOAT", "labels": "FLOAT"},
        "shapes": {"dets": [1, 300, 4], "labels": [1, 300, 91]},
        "variant_claim": "LW-DETR-Large (filename supplied by user)",
        "source": "user-supplied ONNX bundle copied from Cloud 249",
        "preprocessing_reference": {
            "source": "https://github.com/Atten4Vis/LW-DETR/blob/main/deploy/export.py",
            "color": "RGB", "resize": [640, 640], "resize_policy": "stretch",
            "tensor": "float32 NCHW, divide pixels by 255, ImageNet normalize",
            "mean": [.485, .456, .406], "std": [.229, .224, .225]},
        "decoder_contract": {
            "source": "https://github.com/Atten4Vis/LW-DETR/blob/main/models/lwdetr.py",
            "dets": "normalized cxcywh", "labels": "91 sparse COCO category logits",
            "scores": "sigmoid; query x category top-k; map valid COCO IDs to contiguous80",
            "label_map_status": "reference_contract_unverified_checkpoint", "numerical_validation": "not_evaluated"}},
    "rtdetrv2_small": {
        # v4 (5/11): dynamic batch export for Triton (the 10/2026 Cloud artifact is batch-1 only); same IO as dfine_x.
        "inputs": {"images": {"dtype": "FLOAT", "shape": ["N", 3, 640, 640]},
                   "orig_target_sizes": {"dtype": "INT64", "shape": ["N", 2]}},
        "output_dtypes": {"labels": "INT64", "boxes": "FLOAT", "scores": "FLOAT"},
        "shapes": {"labels": [1, 300], "boxes": [1, 300, 4], "scores": [1, 300]},
        "variant_claim": "RT-DETRv2-S r18vd COCO, dynamic batch (rtdetrv2_s_640.onnx copied from Cloud 249)",
        "source": "user-supplied ONNX copied from Cloud 249",
        "preprocessing_reference": {
            "source": "https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/references/deploy/rtdetrv2_onnxruntime.py",
            "color": "RGB", "resize": [640, 640], "resize_policy": "stretch",
            "tensor": "float32 NCHW, divide pixels by 255, no mean/std"},
        "decoder_contract": {
            "source": "https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py",
            "boxes": "xyxy pixels of orig_target_sizes", "labels": "contiguous COCO80",
            "scores": "sigmoid, query x class top-300",
            "label_map_status": "boxes_identical_to_sealed_batch1_artifact_on_8_detrac_frames_20261105",
            "numerical_validation": "not_evaluated"}},
    "dfine_x": {
        # Dynamic batch export (deploy model + postprocessor, opset 17).
        "inputs": {"images": {"dtype": "FLOAT", "shape": ["N", 3, 640, 640]},
                   "orig_target_sizes": {"dtype": "INT64", "shape": ["N", 2]}},
        "output_dtypes": {"labels": "INT64", "boxes": "FLOAT", "scores": "FLOAT"},
        "shapes": {"labels": [1, 300], "boxes": [1, 300, 4], "scores": [1, 300]},
        "variant_claim": "D-FINE-X Objects365+COCO (dfine_x_obj2coco.pth, exported on the laptop)",
        "source": "laptop export of the official checkpoint",
        "preprocessing_reference": {
            "source": "https://github.com/Peterande/D-FINE/blob/master/tools/inference/onnx_inf.py",
            "color": "RGB", "resize": [640, 640], "resize_policy": "stretch",
            "tensor": "float32 NCHW, divide pixels by 255, no mean/std"},
        "decoder_contract": {
            "source": "https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/postprocessor.py",
            "boxes": "xyxy pixels of orig_target_sizes", "labels": "contiguous COCO80 (no COCO id remap)",
            "scores": "sigmoid, query x class top-300",
            "label_map_status": "contiguous_checked_on_ua_detrac_dumps", "numerical_validation": "not_evaluated"}},
}


def import_candidate(artifact, kind="lwdetr_large"):
    """Record observable contract without inventing checkpoint provenance."""
    from datetime import date
    import onnx
    contract = IMPORT_CONTRACTS[kind]
    artifact = Path(artifact).expanduser().resolve(strict=True)
    manifest_path, checksum_path = Path(str(artifact) + ".json"), artifact.parent / "SHA256SUMS"
    if artifact.suffix != ".onnx" or manifest_path.exists() or checksum_path.exists():
        raise ValueError("choose a .onnx without an existing manifest/SHA256SUMS; overwrite forbidden")
    _check_external_references(artifact, set())
    model = onnx.load(str(artifact), load_external_data=False)
    shapes = probe_onnx(artifact, kind)
    if shapes != contract["shapes"]:
        raise ValueError(f"imported {kind} requires batch1 output shapes {contract['shapes']}")
    initializers = {value.name for value in model.graph.initializer}

    def metadata(value):
        tensor = value.type.tensor_type
        return {"dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                          for dim in tensor.shape.dim]}

    inputs = {value.name: metadata(value) for value in model.graph.input if value.name not in initializers}
    outputs = {value.name: metadata(value) for value in model.graph.output}
    if inputs != contract["inputs"]:
        raise ValueError(f"imported {kind} requires inputs {contract['inputs']}")
    if {name: value["dtype"] for name, value in outputs.items()} != contract["output_dtypes"]:
        raise ValueError(f"imported {kind} requires outputs {contract['output_dtypes']}")
    digest = sha256(artifact)
    data = {
        "schema_version": 2, "manifest_type": "imported_candidate", "kind": kind,
        "model_id": candidate_model_id(kind, digest),
        "variant_claim": contract["variant_claim"], "created_on": date.today().isoformat(),
        "artifact": artifact.name, "sha256": digest, "size_bytes": artifact.stat().st_size,
        "external_data": [], "producer": model.producer_name,
        "opsets": {value.domain or "ai.onnx": value.version for value in model.opset_import},
        "input_size": 640, "input_layout": "nchw", "execution_batch_size": 1,
        "inputs": inputs, "outputs": outputs, "output_shapes": shapes,
        "preprocessing_reference": contract["preprocessing_reference"],
        "decoder_contract": contract["decoder_contract"],
        "provenance": {"source": contract["source"],
                       "source_repository": SOURCES[kind][0],
                       "source_revision": None, "checkpoint_sha256": None,
                       "export_config": None, "license": None, "verified": False},
        "validation": {"file_checksums": "verified", "onnx_checker": "passed_locally",
                       "local_inference": "cpu_synthetic_tensor_contract_smoke_passed",
                       "adapter_cuda_inference": "not_run", "quality": "not_evaluated",
                       "pipeline_benchmark": "not_run"},
        "admission": {"status": "candidate_only", "production_ready": False,
                      "adapter_ready": False, "note": "Explicit opt-in required; CUDA/box/quality gates pending."}}
    with manifest_path.open("x", encoding="utf-8") as target:
        json.dump(data, target, indent=2, allow_nan=False)
        target.write("\n")
    with checksum_path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(f"{digest}  {artifact.name}\n{sha256(manifest_path)}  {manifest_path.name}\n")
    print(json.dumps({"manifest": str(manifest_path), "model_id": data["model_id"],
                      "gate": "CPU_CONTRACT_ONLY_CUDA_PENDING"}))
    return data


def seal(artifact, kind, *, repo, checkpoint, reference_onnx=None, layout="nchw", export_evidence=None):
    artifact, repo, checkpoint = (Path(value).expanduser().resolve(strict=True) for value in (artifact, repo, checkpoint))
    if kind not in MODEL_IDS or layout not in ({"nhwc", "nchw"} if kind == "rtmdet_small" else {"nchw"}):
        raise ValueError("unsupported model kind/input layout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip())
    if dirty:
        raise ValueError("upstream tracked source must be clean for export provenance")
    url, config = SOURCES[kind]
    if not (repo / config).is_file() or not (repo / "LICENSE").is_file():
        raise ValueError("upstream config/license not found")
    if artifact.suffix == ".dlc":
        if kind != "rtmdet_small" or not reference_onnx:
            raise ValueError("RTMDet DLC requires its source --reference-onnx")
        reference = Path(reference_onnx).resolve(strict=True)
        reference_manifest = json.loads(Path(str(reference) + ".json").read_text(encoding="utf-8"))
        if (reference_manifest.get("kind") != kind or reference_manifest.get("model_id") != MODEL_IDS[kind]
                or reference_manifest.get("sha256") != sha256(reference)
                or reference_manifest.get("checkpoint_sha256") != sha256(checkpoint)
                or reference_manifest.get("source_revision") != revision):
            raise ValueError("DLC reference ONNX provenance/checksum mismatch")
        shapes = probe_onnx(reference, kind)
    else:
        if artifact.suffix != ".onnx":
            raise ValueError("expected ONNX or RTMDet DLC")
        shapes = probe_onnx(artifact, kind)
    data = {"schema_version": 1, "kind": kind, "model_id": MODEL_IDS[kind],
            "sha256": sha256(artifact), "input_size": 640, "input_layout": layout,
            "output_shapes": shapes, "source_url": url, "source_revision": revision,
            "source_config": config, "config_sha256": sha256(repo / config),
            "checkpoint_sha256": sha256(checkpoint), "checkpoint_name": checkpoint.name,
            "license": "Apache-2.0", "license_sha256": sha256(repo / "LICENSE"),
            "label_map": "coco80_contiguous", "seal_probe": "onnxruntime_cpu_batch1",
            "hardware_validated": False, "quality_status": "NOT_EVALUATED"}
    if reference_onnx:
        data["reference_onnx_sha256"] = sha256(Path(reference_onnx))
    if export_evidence:
        data["export_evidence"] = export_evidence
    output = Path(str(artifact) + ".json")
    with output.open("x", encoding="utf-8") as target:
        json.dump(data, target, indent=2, allow_nan=False)
    print(f"SEALED (hardware pending): {output}")
    return data


def verify(artifact, kind, image, report_path, frames, *, allow_imported_candidate=False,
           video=None, frame_index=0, overlay_path=None):
    import cv2
    import os
    from inference.cloud_detr import RtdetrV2SmallAdapter, LwDetrLargeAdapter, DfineXAdapter
    from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter
    if frames < 1 or frames > 1000:
        raise ValueError("smoke frames must be 1..1000")
    classes = {"rtdetrv2_small": RtdetrV2SmallAdapter, "lwdetr_large": LwDetrLargeAdapter,
               "dfine_x": DfineXAdapter, "rtmdet_small": RtmdetSmallSnpeAdapter}
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError("choose a new report; overwrite forbidden")
    overlay_path = Path(overlay_path) if overlay_path else None
    if overlay_path and (overlay_path.suffix.lower() != ".png" or overlay_path.exists()):
        raise ValueError("choose a new .png overlay; overwrite forbidden")
    if frame_index < 0 or (image is None) == (video is None):
        raise ValueError("provide exactly one of image/video and a nonnegative frame index")
    if video is not None:
        capture = cv2.VideoCapture(str(video))
        try:
            if frame_index and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError("could not seek video frame")
            ok, pixels = capture.read()
            if not ok:
                raise ValueError("could not read representative video frame")
        finally:
            capture.release()
    else:
        pixels = cv2.imread(str(image))
    if pixels is None or pixels.shape[0] == pixels.shape[1]:
        raise ValueError("provide a non-square representative image to verify original-size mapping")
    os.environ["MODEL_ORT_PROFILE"] = "1"
    if allow_imported_candidate and kind not in {"rtdetrv2_small", "lwdetr_large", "dfine_x"}:
        raise ValueError("imported candidate verification supports RT-DETRv2-S, LW-DETR-L and D-FINE-X only")
    kwargs = {"allow_imported_candidate": True} if allow_imported_candidate else {}
    adapter = classes[kind](str(artifact), **kwargs)
    detections = []
    for _ in range(frames):
        boxes, elapsed = adapter.infer(pixels)
        detections.append({"boxes": len(boxes), "adapter_ms": elapsed})
    evidence = {"model_id": adapter.model_id, "sha256": sha256(Path(artifact)),
                "runtime": adapter.runtime, "frames": frames, "source_shape": list(pixels.shape),
                "settings": adapter.settings, "smoke": detections, "quality_status": "NOT_EVALUATED",
                "tracking": False, "pipeline_benchmark": False, "parity": "PENDING"}
    evidence["source"] = {"image": str(Path(image).resolve()) if image else None,
                          "video": str(Path(video).resolve()) if video else None, "frame_index": frame_index}
    if kind != "rtmdet_small":
        profile_path = adapter.session.end_profiling()
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        kernels = [e.get("args", {}) for e in profile if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
        evidence["profile"] = str(Path(profile_path).resolve())
        evidence["cuda_kernels"] = sum(e["provider"] == "CUDAExecutionProvider" for e in kernels)
        evidence["cpu_ops"] = sorted(set(e.get("op_name", "unknown") for e in kernels if e["provider"] == "CPUExecutionProvider"))
        evidence["cuda_active"] = evidence["cuda_kernels"] > 0
        evidence["runtime_gate"] = "REVIEW_CPU_OPS" if evidence["cpu_ops"] else "PASS"
        if not evidence["cuda_active"]:
            evidence["runtime_gate"] = "FAIL_NO_CUDA_KERNELS"
    else:
        evidence["session_mode"] = adapter.session.mode
        evidence["runtime_gate"] = "PENDING_DSP_PROFILING_AND_PARITY"
    # Preserve the actual outputs for upstream/ONNX parity, separate from performance.
    from dataclasses import asdict
    evidence["last_boxes"] = [asdict(box) for box in boxes]
    if overlay_path:
        from PIL import Image, ImageDraw
        canvas = Image.fromarray(pixels[..., ::-1])
        draw = ImageDraw.Draw(canvas)
        height, width = pixels.shape[:2]
        for box in boxes:
            bounds = (box.left * width, box.top * height, box.right * width, box.bottom * height)
            draw.rectangle(bounds, outline=(0, 255, 0), width=2)
            draw.text(bounds[:2], f"{box.class_name} {box.confidence:.2f}", fill=(255, 255, 0))
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        with overlay_path.open("xb") as target:
            canvas.save(target, format="PNG")
        evidence["overlay"] = str(overlay_path.resolve())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as target:
        json.dump(evidence, target, indent=2, allow_nan=False)
    print(json.dumps({"report": str(report_path.resolve()), "runtime_gate": evidence["runtime_gate"],
                      "frames": frames, "detections": len(boxes)}))
    return 1 if evidence["runtime_gate"].startswith("FAIL") else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seal", "import", "verify"])
    parser.add_argument("--kind", required=True, choices=tuple(MODEL_IDS))
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-onnx", type=Path)
    parser.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path, help="read one non-square frame using OpenCV; no system ffmpeg required")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--overlay", type=Path, help="save a new .png with decoded boxes for visual review")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--allow-imported-candidate", action="store_true",
                        help="test imported RT-DETRv2-S/LW-DETR-L/D-FINE-X schema2 candidate; not production admission")
    args = parser.parse_args()
    if args.command == "import":
        if args.kind not in IMPORT_CONTRACTS:
            parser.error("import supports the self-contained LW-DETR-L and D-FINE-X exports")
        import_candidate(args.artifact, args.kind)
        return 0
    if args.command == "seal":
        if args.allow_imported_candidate:
            parser.error("--allow-imported-candidate is for verify, not provenance sealing")
        if not args.repo or not args.checkpoint:
            parser.error("seal requires --repo and --checkpoint")
        seal(args.artifact, args.kind, repo=args.repo, checkpoint=args.checkpoint,
             reference_onnx=args.reference_onnx, layout=args.layout)
        return 0
    if bool(args.image) == bool(args.video) or not args.report:
        parser.error("verify requires exactly one of --image/--video, and --report")
    return verify(args.artifact, args.kind, args.image, args.report, args.frames,
                  allow_imported_candidate=args.allow_imported_candidate, video=args.video,
                  frame_index=args.frame_index, overlay_path=args.overlay)


if __name__ == "__main__":
    raise SystemExit(main())
