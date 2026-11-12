"""Real-frame calibration bound to a verified export; SNPE NHWC requires IO confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.model_artifact import sha256
from scripts.rebuild_rtmdet_small import CHECKPOINT_SHA256, REVISION


def calibration_tensor(nchw, layout):
    import numpy as np
    if (nchw.shape != (1, 3, 640, 640) or nchw.dtype != np.float32
            or not np.isfinite(nchw).all() or layout not in ("nchw", "nhwc")):
        raise ValueError("calibration tensor contract failed")
    return np.ascontiguousarray(nchw if layout == "nchw" else nchw.transpose(0, 2, 3, 1))


def runtime_paths(root, names):
    if (not root or not root.startswith("/") or root.startswith("//")
            or any(c.isspace() for c in root) or "\\" in root
            or ".." in PurePosixPath(root).parts or PurePosixPath(root).as_posix() != root):
        raise ValueError("runtime input root must be a canonical absolute Linux path without whitespace")
    if any(PurePosixPath(name).name != name or name in (".", "..") or "\\" in name
           or any(c.isspace() for c in name) for name in names):
        raise ValueError("unsafe calibration filename")
    return [str(PurePosixPath(root) / name) for name in names]


def checked_reference(folder):
    folder = Path(folder).resolve(strict=True)
    artifact = folder / "rtmdet_small_float.onnx"
    report = json.loads((folder / "rebuild_report.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(str(artifact) + ".json").read_text(encoding="utf-8"))
    evidence = report.get("evidence", {})
    if (report.get("export_gate") != "PASS" or report.get("source_revision") != REVISION
            or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or report.get("artifact_sha256") != sha256(artifact)
            or manifest.get("sha256") != report["artifact_sha256"]
            or manifest.get("source_revision") != REVISION
            or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or manifest.get("input_layout") != "nchw"
            or manifest.get("input_size") != 640
            or manifest.get("output_shapes") != {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]}
            or manifest.get("export_evidence") != evidence
            or evidence.get("preprocess_vs_mmdetection") != "PASS"
            or evidence.get("torch_onnx_raw_parity") != "PASS"
            or evidence.get("onnx_decoder_vs_mmdetection", {}).get("gate") != "PASS"):
        raise ValueError("reference export/source/manifest/checksum/parity gate failed")
    return artifact, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--layout", choices=("nchw", "nhwc"), default="nchw")
    parser.add_argument("--snpe-nhwc-io-confirmed", action="store_true",
                        help="only after DLC info confirms input [1,640,640,3] and decoded boxes/scores outputs")
    parser.add_argument("--runtime-input-root", help="absolute Linux calibration directory used by input_list.txt")
    parser.add_argument("--package", type=Path, help="optional new ZIP of calibration files with SHA256SUMS")
    args = parser.parse_args()
    if not 10 <= args.samples <= 1000 or args.output_dir.exists():
        parser.error("10..1000 samples; output directory must be new")
    if args.layout == "nhwc" and not args.snpe_nhwc_io_confirmed:
        parser.error("confirm converted DLC IO before preparing NHWC calibration")
    if args.package and args.package.exists():
        parser.error("package must be new")
    if args.runtime_input_root:
        runtime_paths(args.runtime_input_root, [])
    artifact, report = checked_reference(args.reference_dir)
    import cv2
    import numpy as np
    from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter
    cv2.setNumThreads(1)
    video = args.video.resolve(strict=True)
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or total < args.samples:
            raise ValueError("video cannot supply requested distinct samples")
        output = args.output_dir.resolve()
        output.mkdir(parents=True)
        adapter = object.__new__(RtmdetSmallSnpeAdapter)
        adapter.manifest = {"input_layout": "nchw"}
        samples = []
        for frame_id in np.linspace(0, total - 1, args.samples, dtype=np.int64):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"cannot decode calibration frame {frame_id}")
            nchw = adapter.prepare_frame(image).tensor
            tensor = calibration_tensor(nchw, args.layout)
            name = f"frame_{frame_id:06d}.raw"
            with (output / name).open("xb") as stream:
                stream.write(tensor.tobytes())
            samples.append({"frame_id": int(frame_id), "file": name, "size_bytes": (output / name).stat().st_size,
                            "tensor_sha256": sha256(output / name),
                            "source_nchw_sha256": hashlib.sha256(nchw.tobytes()).hexdigest(),
                            "decoded_bgr_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "source_image_shape": list(image.shape)})
        metadata = {"schema_version": 1, "source_video_sha256": sha256(video),
                    "reference_artifact_sha256": sha256(artifact),
                    "reference_report_sha256": sha256(args.reference_dir / "rebuild_report.json"),
                    "checkpoint_sha256": report["checkpoint_sha256"], "source_revision": report["source_revision"],
                    "input_name": "input", "input_layout": args.layout, "dtype": "float32",
                    "byte_order": sys.byteorder, "shape": list(tensor.shape),
                    "layout_transform": "transpose(0,2,3,1); no arithmetic" if args.layout == "nhwc" else "none",
                    "snpe_nhwc_io_confirmed": args.snpe_nhwc_io_confirmed,
                    "sample_count": len(samples), "samples": samples,
                    "preprocessing_evidence": report["evidence"],
                    "quantization_gate": "NOT_RUN", "quality_status": "NOT_EVALUATED"}
        with (output / "calibration.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
        if args.runtime_input_root:
            paths = runtime_paths(args.runtime_input_root, [sample["file"] for sample in samples])
            with (output / "input_list.txt").open("x", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(paths) + "\n")
        files = sorted(output.iterdir())
        with (output / "SHA256SUMS").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("".join(f"{sha256(path)}  {path.name}\n" for path in files))
        package_hash = None
        if args.package:
            with zipfile.ZipFile(args.package, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for path in sorted(output.iterdir()):
                    archive.write(path, path.name)
            with zipfile.ZipFile(args.package) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError(f"package CRC failed: {bad}")
            package_hash = sha256(args.package)
        print(json.dumps({"output": str(output), "samples": len(samples),
                          "input_layout": args.layout, "package_sha256": package_hash,
                          "calibration_gate": "REAL_INPUTS_PREPARED", "quantization_gate": "NOT_RUN"}))
    finally:
        cap.release()


if __name__ == "__main__":
    main()
