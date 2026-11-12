"""Offline candidate video trial; one SNPE process, no production admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo11s_contract import SHAPES, prepare, decode, digest


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def stats(values):
    if not values:
        return None
    return {"count": len(values), "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def completion_timing(times, warmup):
    # Filesystem completion cadence, not SDK execute time or live-pipeline FPS.
    if warmup < 1 or len(times) <= warmup:
        raise ValueError("warmup and measured outputs required")
    measured = np.diff(np.asarray(times[warmup - 1:], dtype=np.int64)) / 1e6
    if not (measured > 0).all():
        return {"fps": None, "interval_ms": None, "reason": "non-increasing/coarse output timestamps"}
    return {"fps": float(1000 / np.mean(measured)), "interval_ms": stats(measured.tolist()),
            "scope": "SNPE output-file completion; includes input/output IO; startup and warmup excluded"}


def writer(path, fps, width, height):
    value = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not value.isOpened():
        raise RuntimeError("OpenCV MP4 encoder unavailable; video not silently skipped")
    return value


def draw(image, boxes, index, nms):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for box in boxes:
        a = (round(box.left * width), round(box.top * height))
        b = (round(box.right * width), round(box.bottom * height))
        cv2.rectangle(canvas, a, b, (0, 220, 0), 2)
        cv2.putText(canvas, f"{box.class_name} {box.confidence:.2f}", (a[0], max(15, a[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1)
    cv2.putText(canvas, f"candidate / frame {index} / NMS {nms:.2f} / NOT QUALITY ADMITTED",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)
    return canvas


def clip(args):
    if args.output.exists():
        raise FileExistsError("choose a new clip path")
    cap = cv2.VideoCapture(str(args.video.resolve(strict=True)))
    encoder = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("cannot open source video/fps")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        for index in range(args.count):
            ok, image = cap.read()
            if not ok:
                raise ValueError("source does not contain requested clip")
            if encoder is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                encoder = writer(args.output, fps, image.shape[1], image.shape[0])
            encoder.write(image)
    finally:
        cap.release()
        if encoder is not None:
            encoder.release()
    check = cv2.VideoCapture(str(args.output))
    count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if count != args.count:
        raise ValueError("encoded clip frame count mismatch")
    print(json.dumps({"clip": str(args.output.resolve()), "frames": count, "sha256": digest(args.output)}))
    return 0


def trial(args):
    cv2.setNumThreads(1)
    if args.output_dir.exists():
        raise FileExistsError("choose a new trial output directory")
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    report = {"status": "STARTED", "diagnostic_only": True, "production_ready": False,
              "quality_status": "NOT_EVALUATED", "nms_iou": args.nms_iou, "confidence": .1,
              "warmup_frames": args.warmup, "measured_frames": args.frames,
              "video_mode": "offline render at source playback FPS; not live throughput",
              "video_sha256": digest(args.video), "stages_ms": {}, "completed": 0}
    cap, render_cap, encoder = None, None, None
    try:
        session = None
        if args.local_float_onnx:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(args.local_float_onnx), options, providers=["CPUExecutionProvider"])
            report["runtime"] = "LOCAL_FLOAT_CONTROL_NOT_NPU"
            report["artifact_sha256"] = digest(args.local_float_onnx)
        else:
            artifact = args.artifact_root.resolve(strict=True)
            dlc = artifact / "yolo11s320_w8a8_qcs6490.dlc"
            marker = json.loads((artifact / "diagnostic.json").read_text(encoding="utf-8"))
            if (marker.get("diagnostic_only") is not True or marker.get("production_ready") is not False
                    or digest(dlc) != marker["prepared_candidate_sha256"]):
                raise ValueError("candidate diagnostic marker/checksum mismatch")
            info = (artifact / "qcs6490_info.log").read_text(errors="replace")
            graph = re.findall(r"Info of graph\s*:\s*([\w.-]+)", info)
            if not graph:
                graph = re.findall(r"Graph Name\s*:\s*([\w.-]+)", info)
            if len(set(graph)) != 1 or "HTP_V68_QCS6490_2MB" not in info:
                raise ValueError("expected one QCS6490 graph")
            sdk = args.sdk.resolve(strict=True)
            bins = sdk / "bin/aarch64-oe-linux-gcc11.2"
            runner = bins / "snpe-net-run"
            if not runner.is_file():
                raise FileNotFoundError(str(runner))
            report.update(runtime="SNPE_DSP_HTP_NO_CPU_FALLBACK", artifact_sha256=digest(dlc),
                          prior_cpu_gate="FAIL", prior_npu_gate="FAIL")
        total = args.frames + args.warmup
        # Float NHWC input + float output files; retain them for reproducible review.
        required = total * (320 * 320 * 3 * 4 + sum(int(np.prod(s)) * 4 for s in SHAPES.values())) + 128 * 1024**2
        if shutil.disk_usage(root).free < required:
            raise OSError(f"need at least {required / 1024**2:.0f} MiB free for trial files")
        inputs = root / "inputs"
        outputs = root / "npu_output"
        inputs.mkdir()
        outputs.mkdir()
        cap = cv2.VideoCapture(str(args.video.resolve(strict=True)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("invalid input video/fps")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        prepared_frames, preprocessing, input_write, inference = [], [], [], []
        for index in range(total):
            ok, image = cap.read()
            if not ok or image.shape[:2] != (height, width):
                raise ValueError(f"missing/changing video frame {index}")
            started = time.perf_counter()
            prepared = prepare(image)
            elapsed = (time.perf_counter() - started) * 1000
            prepared_frames.append((prepared.width, prepared.height, prepared.gain, prepared.pad_x, prepared.pad_y))
            started = time.perf_counter()
            prepared.tensor.transpose(0, 2, 3, 1).astype("<f4").tofile(inputs / f"frame_{index:06d}.raw")
            written = (time.perf_counter() - started) * 1000
            if index >= args.warmup:
                preprocessing.append(elapsed)
                input_write.append(written)
            if session is not None:
                started = time.perf_counter()
                values = session.run(list(SHAPES), {"images": prepared.tensor})
                elapsed = (time.perf_counter() - started) * 1000
                folder = outputs / f"Result_{index}"
                folder.mkdir()
                for name, value in zip(SHAPES, values):
                    value.astype("<f4").tofile(folder / f"{name}.raw")
                if index >= args.warmup:
                    inference.append(elapsed)
            if index % 60 == 0:
                print(json.dumps({"prepared": index + 1, "total": total}), flush=True)
        cap.release()
        cap = None
        if session is None:
            listing = root / "input_list.txt"
            listing.write_text("".join(str(inputs / f"frame_{i:06d}.raw") + "\n" for i in range(total)))
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{sdk}/lib/aarch64-oe-linux-gcc11.2:{sys.prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            env["ADSP_LIBRARY_PATH"] = f"{sdk}/lib/hexagon-v68/unsigned;{sdk}/lib/aarch64-oe-linux-gcc11.2;/usr/lib/rfsa/adsp;/dsp;/opt"
            env["PYTHONPATH"] = f"{sdk}/lib/python:{env.get('PYTHONPATH', '')}"
            command = [str(runner), "--container", str(dlc), "--input_list", str(listing),
                       "--output_dir", str(outputs), "--use_dsp", "--profiling_level", "basic",
                       "--set_output_tensors", f"{graph[0]} boxes,scores"]
            started = time.perf_counter()
            with (root / "snpe_run.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            report["process_wall_s_including_startup_warmup_io"] = time.perf_counter() - started
            report["return_code"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError("SNPE DSP failed; see snpe_run.log")
            diag = bins / "snpe-diagview"
            diag_log = outputs / "SNPEDiag_0.log"
            if diag.is_file() and diag_log.is_file():
                try:
                    with (root / "snpe_profile.txt").open("w") as log:
                        completed = subprocess.run([str(diag), "--input_log", str(diag_log),
                                                    "--output", str(root / "snpe_profile.csv")],
                                                   env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90)
                    report["profiler_export_return_code"] = completed.returncode
                except (OSError, subprocess.TimeoutExpired) as error:
                    report["profiler_export_error"] = str(error)
            report["profiler_scope"] = "SDK aggregate may include warmup; retained, not reported as measured-only execute latency"
            times = [max((outputs / f"Result_{i}" / f"{k}.raw").stat().st_mtime_ns for k in SHAPES)
                     for i in range(total)]
            report["snpe_output_completion"] = completion_timing(times, args.warmup)
        render_cap = cv2.VideoCapture(str(args.video))
        encoder = writer(root / "annotated.mp4", fps, width, height)
        postprocess, counts = [], []
        with (root / "detections.jsonl").open("w", encoding="utf-8") as detection_log:
            for index in range(total):
                ok, image = render_cap.read()
                if not ok:
                    raise ValueError("source video replay failed")
                if index < args.warmup:
                    continue
                values = {}
                for name, shape in SHAPES.items():
                    path = outputs / f"Result_{index}" / f"{name}.raw"
                    if path.stat().st_size != int(np.prod(shape)) * 4:
                        raise ValueError(f"wrong byte count: {path}")
                    values[name] = np.fromfile(path, dtype="<f4").reshape(shape)
                from types import SimpleNamespace
                w, h, gain, px, py = prepared_frames[index]
                prepared = SimpleNamespace(width=w, height=h, gain=gain, pad_x=px, pad_y=py)
                started = time.perf_counter()
                detections = decode(values, prepared, nms_iou=args.nms_iou)
                postprocess.append((time.perf_counter() - started) * 1000)
                counts.append(len(detections))
                detection_log.write(json.dumps({"frame": index, "detections": [asdict(d) for d in detections]}) + "\n")
                canvas = draw(image, detections, index, args.nms_iou)
                encoder.write(canvas)
                if index - args.warmup in (0, args.frames // 2, args.frames - 1):
                    if not cv2.imwrite(str(root / f"preview_{index}.png"), canvas):
                        raise OSError("preview write failed")
                report["completed"] += 1
        encoder.release()
        encoder = None
        check = cv2.VideoCapture(str(root / "annotated.mp4"))
        count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        check.release()
        if count != args.frames:
            raise ValueError("rendered video frame count mismatch")
        report.update(status="VIDEO_TRIAL_COMPLETE_NOT_QUALITY_ADMITTED", detections_total=sum(counts),
                      playback_fps=fps, rendered_frames=count)
        report["stages_ms"] = {"preprocess": stats(preprocessing), "input_file_write": stats(input_write),
                               "postprocess": stats(postprocess), "inference_execute": stats(inference)}
        if session is None:
            report["inference_execute_note"] = "null: direct measured-only SDK execute time not parsed; use profiler, do not substitute process walltime"
    except Exception as error:
        report.update(status="VIDEO_TRIAL_FAILED", error=str(error))
    finally:
        for value in (cap, render_cap, encoder):
            if value is not None:
                value.release()
        save_json(root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(root / "report.json"),
                      "video": str(root / "annotated.mp4"), "completed": report["completed"]}), flush=True)
    return 0 if report["status"] == "VIDEO_TRIAL_COMPLETE_NOT_QUALITY_ADMITTED" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("clip")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=100)
    p.add_argument("--count", type=int, default=330)
    p.set_defaults(run=clip)
    p = sub.add_parser("run")
    p.add_argument("--video", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--artifact-root", type=Path)
    group.add_argument("--local-float-onnx", type=Path, help="local control test ONLY, never NPU")
    p.add_argument("--sdk", type=Path, default=Path("/home/ubuntu/sigm/qairt/2.43.0.260128"))
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--nms-iou", type=float, default=.5)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(run=trial)
    args = parser.parse_args()
    if args.command == "clip" and (args.start < 0 or not 1 <= args.count <= 1000):
        parser.error("invalid clip interval")
    if args.command == "run" and (not 1 <= args.frames <= 600 or not 1 <= args.warmup <= 100
                                   or not 0 < args.nms_iou <= 1):
        parser.error("invalid trial counts/NMS")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
