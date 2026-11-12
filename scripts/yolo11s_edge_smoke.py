"""Persistent adapter smoke before live; never requires Cloud or claims quality PASS."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlc", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 1000 or not 0 <= args.warmup <= 100:
        parser.error("frames must be1..1000, warmup0..100")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "diagnostic_only": True, "production_ready": False,
              "quality_status": "NOT_EVALUATED", "parity_gate": "FAIL", "completed": 0,
              "runtime_requested": "DSP persistent daemon", "cpu_fallback_enabled": False,
              "tracking": False, "full_pipeline_gate": "NOT_RUN", "frames": [],
              "note": "Adapter contract smoke only; not mAP, live FPS or quality admission."}
    capture = detector = None
    try:
        import cv2
        from inference.yolo11s_snpe import Yolo11sSnpeAdapter
        from yolo11s_contract import digest, overlay
        report["video_sha256"] = digest(args.video)
        detector = Yolo11sSnpeAdapter(args.dlc, allow_diagnostic=args.allow_diagnostic)
        report["settings"] = detector.settings
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise ValueError("video could not be opened")
        for index in range(args.warmup + args.frames):
            ok, image = capture.read()
            if not ok:
                raise ValueError("video ended before requested frame count")
            started = perf_counter()
            prepared = detector.prepare_frame(image)
            boxes, elapsed, stages = detector.infer_prepared(prepared)
            if index >= args.warmup:
                entry = {"source_frame": index, "detections": [asdict(b) for b in boxes],
                         "adapter_ms": elapsed, "stages_ms": stages,
                         "adapter_call_ms": (perf_counter() - started) * 1000}
                report["frames"].append(entry)
                report["completed"] += 1
                if report["completed"] <= 3:
                    overlay(image, boxes, args.out / f"frame_{index:06d}.png")
        report["status"] = "PERSISTENT_ADAPTER_CONTRACT_PASS_QUALITY_NOT_ADMITTED"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.session.close()  # Shared daemon intentionally retained, not killed.
        with (args.out / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "completed": report["completed"],
                      "report": str(args.out / "report.json"), "error": report.get("error")}))
    return 0 if report["completed"] == args.frames and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
