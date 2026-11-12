"""Three-image Float CPU parity and standalone timing; no transport/DRL admission."""
import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import Box
from inference.rtmdet_float_cpu import RtmdetSmallFloatCpuAdapter
from inference.model_artifact import sha256
from scripts.rebuild_rtmdet_small import compare_detections


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=root / "float_artifact/rtmdet_small_float.onnx")
    parser.add_argument("--references", type=Path, default=root / "float_references")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be 1..100")
    # Verify the source inventory; the adapter separately checks the artifact identity.
    manifest_file = root / "RELEASE_MANIFEST.json"
    if manifest_file.is_file():
        for name, wanted in json.loads(manifest_file.read_text()).items():
            path = root / name
            if path.resolve().is_relative_to(root.resolve()) is False or sha256(path) != wanted:
                raise ValueError("Source release checksum mismatch: " + name)
    work = Path(tempfile.mkdtemp(prefix="cpu_smoke_", dir=args.artifact.parent))
    report = {"runtime": "onnx_cpu", "placement": "edge", "precision": "float32",
              "architecture": platform.machine(), "cpu_threads": args.threads,
              "quality": "NOT_EVALUATED", "pipeline_benchmark": "NOT_RUN", "frames": [],
              "gate": "FAIL", "production_ready": False}
    print("SMOKE_DIR=" + str(work), flush=True)
    times = []
    try:
        cv2.setNumThreads(1)
        start = perf_counter()
        adapter = RtmdetSmallFloatCpuAdapter(str(args.artifact), threads=args.threads)
        report["load_ms"] = (perf_counter() - start) * 1000
        report["model_id"] = adapter.model_id
        report["providers"] = adapter.session.get_providers()
        for number in (120, 600, 1200):
            folder = args.references / f"frame_{number:06d}"
            hashes = json.loads((folder / "REFERENCE_SHA256.json").read_text())
            if set(hashes) != {"source.png", "reference_input.npy", "rebuild_report.json"}:
                raise ValueError("Reference inventory mismatch")
            for name, digest in hashes.items():
                if Path(name).name != name or sha256(folder / name) != digest:
                    raise ValueError("Reference checksum mismatch")
            reference = json.loads((folder / "rebuild_report.json").read_text())
            if (reference["artifact_sha256"] != adapter.manifest["sha256"]
                    or reference["export_gate"] != "PASS"
                    or reference["oracle_img_shape"] != [640, 640]):
                raise ValueError("Reference/model identity mismatch")
            image = cv2.imread(str(folder / "source.png"))
            if image is None:
                raise ValueError("Reference image missing")
            prepared = adapter.prepare_frame(image)
            tensor = np.load(folder / "reference_input.npy", allow_pickle=False)
            np.testing.assert_array_equal(prepared.tensor, tensor)
            expected = [Box(**box) for box in reference["boxes"]]
            adapter.infer(image)  # warmup excluded from timings
            for _ in range(args.repeats):
                boxes, elapsed, stages = adapter.infer_prepared(adapter.prepare_frame(image))
                parity = compare_detections(expected, boxes)
                times.append(elapsed)
            row = {"frame": number, "parity": parity, "detections": len(boxes), "stages_ms": stages}
            report["frames"].append(row)
            print(json.dumps(row), flush=True)
            canvas = image.copy()
            h, w = canvas.shape[:2]
            for box in boxes:
                p1, p2 = (round(box.left*w), round(box.top*h)), (round(box.right*w), round(box.bottom*h))
                cv2.rectangle(canvas, p1, p2, (0, 210, 0), 2)
                cv2.putText(canvas, f"{box.class_name} {box.confidence:.2f}", (p1[0], max(15, p1[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 210, 0), 1)
            if not cv2.imwrite(str(work / f"frame_{number:06d}_boxes.png"), canvas):
                raise ValueError("Overlay write failed")
        report.update(gate="PASS", adapter_mean_ms=float(np.mean(times)),
                      adapter_p95_ms=float(np.percentile(times, 95)),
                      standalone_fps=1000 / float(np.mean(times)), measured_calls=len(times))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (work / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"report": str(work / "report.json"), "gate": report["gate"],
                      "runtime": report["runtime"], "standalone_fps": report.get("standalone_fps"),
                      "error": report.get("error")}), flush=True)
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
