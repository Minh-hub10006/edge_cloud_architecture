"""One bounded YOLO11s320 SNPE W8/A8 candidate build and Edge NPU parity gate."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yolo11s_contract import Detection as Box, SHAPES, agreement, decode, digest, overlay, verify_inventory

SDK_CLOUD = Path("/home/stu0/cloud_sdks/rtmdet_snpe243.atdhop/2.43.0.260128")
SDK_EDGE = Path("/home/ubuntu/sigm/qairt/2.43.0.260128")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run(command, log, env, timeout=600):
    result = subprocess.run([str(v) for v in command], env=env, capture_output=True, text=True, timeout=timeout)
    Path(log).write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8", errors="replace")
    if result.returncode:
        tail = (result.stdout + "\n" + result.stderr)[-1800:]
        raise RuntimeError(f"command failed ({result.returncode}); review {log}\n{tail}")
    return result.stdout + result.stderr


def sdk_env(sdk, python_prefix):
    bins = sdk / "bin/x86_64-linux-clang"
    if not bins.is_dir():
        raise FileNotFoundError("SNPE x86_64-linux-clang tools not found: " + str(bins))
    env = os.environ.copy()
    env["PATH"] = f"{bins}:{python_prefix}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{sdk}/lib/x86_64-linux-clang:{python_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{sdk}/lib/python:{env.get('PYTHONPATH', '')}"
    env["SNPE_ROOT"] = str(sdk)
    env["QAIRT_ROOT"] = str(sdk)
    return env, bins


def verify_raw(path, shape):
    path = Path(path)
    count = int(np.prod(shape))
    if path.stat().st_size != count * 4:
        raise ValueError(f"SNPE Float output byte count mismatch: {path}")
    values = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("SNPE returned nonfinite output")
    return values


def graph_name(info):
    names = re.findall(r"Info of graph\s*:\s*([A-Za-z0-9_.-]+)", info)
    if not names:
        names = re.findall(r"Graph Name\s*:\s*([A-Za-z0-9_.-]+)", info)
    if not names:
        raise ValueError("DLC graph name not found; inspect snpe-dlc-info output")
    if len(set(names)) != 1:
        raise ValueError("candidate requires exactly one DLC graph")
    return names[0]


def ensure_new(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("each run needs a fresh output folder: " + str(path))
    path.mkdir(parents=True)
    return path


def load_packet(root):
    root = verify_inventory(root)
    manifest = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != "yolo11s320_candidate" or manifest.get("float_gate") != "PASS"
            or manifest.get("input_size") != 320 or len(manifest.get("calibration_frames", [])) < 60
            or len(manifest.get("reference_frames", [])) != 3):
        raise ValueError("incomplete or unverified YOLO11s320 Float packet")
    return root, manifest


def unpack_report(root):
    references, inputs = [], []
    for index in (120, 600, 1200):
        folder = root / f"frame_{index:06d}"
        record = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        refs = reference_boxes(record)
        refs_float = np.fromfile(folder / "input_nhwc.raw", dtype="<f4")
        if refs_float.size != 320 * 320 * 3 or len(refs) == 0:
            raise ValueError("invalid YOLO Float frame packet")
        references.append((index, folder, record, refs))
        inputs.append(folder / "input_nhwc.raw")
    return references, inputs


def reference_boxes(record):
    required = {field.name for field in fields(Box)}
    result = []
    for item in record["detections"]:
        if set(item) - required - {"track_id", "calibrated_confidence"}:
            raise ValueError("unknown reference detection fields")
        result.append(Box(**{name: item[name] for name in required}))
    return result


def inspect_dlc(bins, sdk, env, dlc, run_dir, label):
    output = run([sys.executable, bins / "snpe-dlc-info", "-i", dlc], run_dir / f"{label}_info.log", env, 90)
    name = graph_name(output)
    normalized = re.sub(r"\s+", "", output)
    expected = ("InputName", "1,320,320,3", "OutputName", "boxes", "scores", "1,2100,4", "1,2100,80")
    if any(term not in normalized for term in expected):
        raise ValueError(f"{label} DLC signature did not match NHWC320 / boxes / scores; inspect info log")
    return name, output


def check_score_encoding(info):
    matches = re.findall(r"scores encoding\s*:\s*bitwidth\s+(\d+),\s*min\s+([-+\d.eE]+),\s*max\s+([-+\d.eE]+),\s*scale\s+([-+\d.eE]+)", info)
    if not matches:
        raise ValueError("quantized scores encoding missing; inspect candidate_info.log")
    for bits, minimum, maximum, scale in matches:
        lo, hi, step = float(minimum), float(maximum), float(scale)
        if (int(bits) != 8 or not np.isfinite([lo, hi, step]).all()
                or lo < -1e-6 or hi > 1.0001 or hi <= lo or step <= 0 or step > 1.0001 / 255):
            raise ValueError("invalid W8A8 probability encoding; stop before NPU preparation")


def runtime(env, bins, dlc, frames, output, log, graph, mode="cpu"):
    listing = output.parent / f"{output.name}_inputs.txt"
    listing.write_text("".join(str(frame[1] / "input_nhwc.raw") + "\n" for frame in frames), encoding="utf-8")
    command = [bins / "snpe-net-run", "--container", dlc, "--input_list", listing,
               "--output_dir", output, "--set_output_tensors", f"{graph} boxes,scores"]
    if mode == "dsp":
        command.extend(["--use_dsp", "--profiling_level", "detailed"])
    run(command, log, env, timeout=900)
    results = []
    for i, (index, _folder, _record, ref) in enumerate(frames if mode == "cpu" else frames):
        result_dir = output / f"Result_{i}"
        values = {name: verify_raw(result_dir / f"{name}.raw", shape) for name, shape in SHAPES.items()}
        # The original image dimensions/letterbox metadata travel in reference.json.
        data = frames[i]
        if mode == "cpu":
            _index, _folder, record, expected = data
        else:
            _index, _folder, record, expected = data
        height, width = record["image_shape"][:2]
        ratio = min(320 / height, 320 / width)
        pad_x = round((320 - width * ratio) / 2 - .1)
        pad_y = round((320 - height * ratio) / 2 - .1)
        class Prepared:
            pass
        prepared = Prepared()
        prepared.width, prepared.height, prepared.gain, prepared.pad_x, prepared.pad_y = width, height, ratio, pad_x, pad_y
        boxes = decode(values, prepared)
        results.append({"frame": index, "reference": expected, "candidate": boxes,
                        "agreement": agreement(expected, boxes), "raw_shapes": {k: list(v.shape) for k, v in values.items()},
                        "raw_outputs": values, "source": _folder / "source.png"})
    return results


def make_review(run_dir, records, results, manifest, report):
    for result in results:
        overlay(__import__("cv2").imread(str(result["source"])), result["candidate"],
                run_dir / f"candidate_{result['frame']}.png")
        overlay(__import__("cv2").imread(str(result["source"])), result["reference"],
                run_dir / f"float_{result['frame']}.png")
        del result["raw_outputs"]
    summary = {"schema_version": 1, "candidate": "YOLO11s320", "model_sha256": manifest["onnx_sha256"],
               "checkpoint_sha256": manifest["checkpoint_sha256"], "quality_status": "NOT_EVALUATED",
               "production_ready": False, "frames": [{"frame": r["frame"], "agreement": r["agreement"]} for r in results],
               **report}
    write_json(run_dir / "report.json", summary)


def cloud(args):
    root, manifest = load_packet(args.bundle)
    refs, input_frames = unpack_report(root)
    run_dir = ensure_new(args.output_dir)
    sdk = args.sdk.resolve(strict=True)
    env, bins = sdk_env(sdk, Path(sys.prefix))
    tools = ("snpe-onnx-to-dlc", "snpe-dlc-quant", "snpe-dlc-info", "snpe-net-run", "snpe-dlc-graph-prepare")
    for tool in tools:
        if not (bins / tool).is_file():
            raise FileNotFoundError("required SDK tool missing: " + str(bins / tool))
    source_onnx = root / "yolo11s320_float.onnx"
    float_dlc = run_dir / "yolo11s320_float.dlc"
    run([sys.executable, bins / "snpe-onnx-to-dlc", "--input_network", source_onnx,
         "--output_path", float_dlc], run_dir / "convert_float.log", env, 900)
    graph, _info = inspect_dlc(bins, sdk, env, float_dlc, run_dir, "float")
    float_result = runtime(env, bins, float_dlc, refs, run_dir / "float_cpu", run_dir / "float_cpu.log", graph)
    float_pass = all(r["agreement"]["gate"] == "PASS" for r in float_result)
    make_review(run_dir, refs, float_result, manifest, {"status": "FLOAT_CPU_PASS" if float_pass else "FLOAT_CPU_FAIL"})
    if not float_pass:
        print(json.dumps({"status": "FLOAT_CPU_GATE_FAILED", "report": str(run_dir / "report.json")})); return 2
    calibration = root / "calibration_nhwc64"
    calibration_frames = sorted(calibration.glob("*.raw"))
    if len(calibration_frames) < 60:
        raise ValueError("verified calibration set has fewer than 60 images")
    calibration_list = run_dir / "calibration_inputs.txt"
    calibration_list.write_text("".join(str(v) + "\n" for v in calibration_frames), encoding="utf-8")
    candidate = run_dir / "yolo11s320_w8a8.dlc"
    quantizer = bins / "snpe-dlc-quant"
    help_text = run([quantizer, "--help"], run_dir / "quantizer_help.log", env, 90)
    required = ("--weights_bitwidth", "--act_bitwidth", "--bias_bitwidth", "--use_per_channel_quantization")
    if any(value not in help_text for value in required):
        raise ValueError("installed native quantizer does not advertise conventional W8/A8 options")
    run([quantizer, "--input_dlc", float_dlc, "--input_list", calibration_list,
         "--output_dlc", candidate, "--weights_bitwidth", "8", "--act_bitwidth", "8",
         "--bias_bitwidth", "32", "--use_per_channel_quantization"], run_dir / "quantize_w8a8.log", env, 1800)
    candidate_graph, _info = inspect_dlc(bins, sdk, env, candidate, run_dir, "candidate")
    check_score_encoding(_info)
    result = runtime(env, bins, candidate, refs, run_dir / "candidate_cpu", run_dir / "candidate_cpu.log", candidate_graph)
    cpu_pass = all(r["agreement"]["gate"] == "PASS" for r in result)
    make_review(run_dir, refs, result, manifest, {"status": "W8A8_CPU_PASS" if cpu_pass else "W8A8_CPU_FAIL",
                                                  "quantization": "SNPE W8/A8 per-channel weights; one bounded attempt",
                                                  "calibration_frames": len(calibration_frames)})
    if not cpu_pass:
        print(json.dumps({"status": "W8A8_CPU_GATE_FAILED", "report": str(run_dir / "report.json")})); return 2
    return prepare_packet(bins, sdk, env, candidate, candidate_graph, run_dir, refs)


def prepare_packet(bins, sdk, env, candidate, candidate_graph, run_dir, refs, diagnostic=None):
    prep_help = run([bins / "snpe-dlc-graph-prepare", "--help"], run_dir / "prepare_help.log", env, 90)
    if "--htp_socs" not in prep_help or "--set_output_tensors" not in prep_help:
        raise ValueError("SDK lacks explicit QCS6490 or graph-output preparation flags")
    cached = run_dir / "yolo11s320_w8a8_qcs6490.dlc"
    run([bins / "snpe-dlc-graph-prepare", "--input_dlc", candidate, "--output_dlc", cached,
         "--htp_socs", "qcs6490", "--set_output_tensors", f"{candidate_graph} boxes,scores"],
        run_dir / "prepare_qcs6490.log", env, 1800)
    _graph, cache_info = inspect_dlc(bins, sdk, env, cached, run_dir, "qcs6490")
    if diagnostic is None:
        check_score_encoding(cache_info)
    if "HTP_V68_QCS6490_2MB" not in cache_info or "boxes" not in cache_info or "scores" not in cache_info:
        raise ValueError("prepared DLC does not contain explicit QCS6490 dual-output cache")
    transfer = run_dir / "edge_packet.zip"
    with zipfile.ZipFile(transfer, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(cached, "edge_packet/yolo11s320_w8a8_qcs6490.dlc")
        archive.write(run_dir / "qcs6490_info.log", "edge_packet/qcs6490_info.log")
        archive.write(run_dir / "prepare_qcs6490.log", "edge_packet/prepare_qcs6490.log")
        for index, folder, _record, _boxes in refs:
            for filename in ("source.png", "input_nhwc.raw", "reference.json"):
                archive.write(folder / filename, f"edge_packet/frame_{index:06d}/{filename}")
        if diagnostic is not None:
            marker = {"diagnostic_only": True, "production_ready": False,
                      "source_candidate_sha256": digest(candidate), "prepared_candidate_sha256": digest(cached),
                      "cpu_report": diagnostic}
            archive.writestr("edge_packet/diagnostic.json", json.dumps(marker, indent=2, allow_nan=False))
    if diagnostic is not None:
        write_json(run_dir / "report.json", {**marker, "status": "NPU_DIAGNOSTIC_PACKET_READY",
                                             "packet_sha256": digest(transfer)})
    print(json.dumps({"status": "NPU_DIAGNOSTIC_PACKET_READY" if diagnostic is not None else "EDGE_PACKET_READY",
                      "diagnostic_only": diagnostic is not None, "production_ready": False,
                      "float_gate": "PASS", "w8a8_cpu_gate": "FAIL" if diagnostic is not None else "PASS",
                      "qcs6490_cache_gate": "PASS", "packet": str(transfer), "packet_sha256": digest(transfer),
                      "candidate": str(cached), "report": str(run_dir / "report.json")}))
    return 0


def prepare_diagnostic(args):
    root, manifest = load_packet(args.bundle)
    refs, _inputs = unpack_report(root)
    previous = args.existing_run.resolve(strict=True)
    report = json.loads((previous / "report.json").read_text(encoding="utf-8"))
    if (report.get("status") != "W8A8_CPU_FAIL" or report.get("model_sha256") != manifest["onnx_sha256"]
            or report.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or [f.get("frame") for f in report.get("frames", [])] != [120, 600, 1200]):
        raise ValueError("diagnostic requires the matching failed W8A8 CPU run")
    candidate = previous / "yolo11s320_w8a8.dlc"
    if not candidate.is_file():
        raise FileNotFoundError("existing W8A8 DLC missing; no re-quantization allowed")
    run_dir = ensure_new(args.output_dir)
    sdk = args.sdk.resolve(strict=True)
    env, bins = sdk_env(sdk, Path(sys.prefix))
    graph, _info = inspect_dlc(bins, sdk, env, candidate, run_dir, "source_candidate")
    return prepare_packet(bins, sdk, env, candidate, graph, run_dir, refs, diagnostic=report)


def diagnostic_marker(archive, opted_in):
    marked = "edge_packet/diagnostic.json" in archive.namelist()
    if marked and not opted_in:
        raise ValueError("failed CPU candidate requires explicit --diagnostic; no production admission")
    if marked:
        marker = json.loads(archive.read("edge_packet/diagnostic.json"))
        if marker.get("diagnostic_only") is not True or marker.get("production_ready") is not False:
            raise ValueError("invalid diagnostic marker")
        return marker
    return None


def edge(args):
    run_dir = ensure_new(args.output_dir)
    packet = args.packet.resolve(strict=True)
    with zipfile.ZipFile(packet) as archive:
        names = archive.namelist()
        marker = diagnostic_marker(archive, args.diagnostic)
        if (any(Path(n).is_absolute() or ".." in Path(n).parts for n in names)
                or len(names) != (13 if marker else 12) or len(set(names)) != len(names)):
            raise ValueError("unsafe or incomplete edge packet")
        archive.extractall(run_dir)
    root = run_dir / "edge_packet"
    dlcs = list(root.glob("*qcs6490.dlc"))
    if len(dlcs) != 1 or not (root / "qcs6490_info.log").is_file():
        raise ValueError("Edge packet missing prepared DLC/info")
    if marker and digest(dlcs[0]) != marker["prepared_candidate_sha256"]:
        raise ValueError("diagnostic prepared DLC checksum mismatch")
    refs = []
    for index in (120, 600, 1200):
        folder = root / f"frame_{index:06d}"
        source = folder / "source.png"
        record_path = folder / "reference.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        data = folder / "input_nhwc.raw"
        refs.append((index, data, record, reference_boxes(record), source))
    sdk = args.sdk.resolve(strict=True)
    bins = sdk / "bin/aarch64-oe-linux-gcc11.2"
    runner = bins / "snpe-net-run"
    if not runner.is_file():
        raise FileNotFoundError("Edge ARM64 SNPE runner not found: " + str(runner))
    input_list = run_dir / "inputs.txt"
    input_list.write_text("".join(str(r[1]) + "\n" for r in refs), encoding="utf-8")
    env = os.environ.copy()
    target_lib = sdk / "lib/aarch64-oe-linux-gcc11.2"
    dsp_lib = sdk / "lib/hexagon-v68/unsigned"
    env["LD_LIBRARY_PATH"] = f"{target_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["ADSP_LIBRARY_PATH"] = ";".join((str(dsp_lib), str(target_lib), "/usr/lib/rfsa/adsp", "/dsp", "/opt"))
    output = run_dir / "npu_output"
    command = [runner, "--container", dlcs[0], "--input_list", input_list,
               "--output_dir", output, "--use_dsp", "--profiling_level", "detailed",
               "--set_output_tensors", f"{graph_name((root / 'qcs6490_info.log').read_text(errors='replace'))} boxes,scores"]
    started = time.perf_counter()
    completed = subprocess.run([str(v) for v in command], env=env, capture_output=True, text=True, timeout=900)
    (run_dir / "npu_run.log").write_text(completed.stdout + completed.stderr, encoding="utf-8", errors="replace")
    report = {"status": "NPU_EXECUTION_FAILED", "runtime_requested": "SNPE DSP/HTP; CPU fallback not enabled",
              "diagnostic_only": marker is not None, "w8a8_cpu_gate": "FAIL" if marker else "PASS",
              "return_code": completed.returncode, "frames": [], "quality_status": "NOT_EVALUATED",
              "production_ready": False, "packet_sha256": digest(packet), "npu_wall_ms_including_process_startup": (time.perf_counter() - started) * 1000}
    if completed.returncode == 0:
        for i, (index, _input, record, expected, source) in enumerate(refs):
            folder = output / f"Result_{i}"
            try:
                values = {name: verify_raw(folder / f"{name}.raw", shape) for name, shape in SHAPES.items()}
            except (ValueError, OSError) as error:
                report["frames"].append({"frame": index, "output_error": str(error)})
                continue
            height, width = record["image_shape"][:2]
            ratio = min(320 / height, 320 / width)
            class Prepared:
                pass
            prepared = Prepared()
            prepared.width, prepared.height, prepared.gain = width, height, ratio
            prepared.pad_x, prepared.pad_y = round((320 - width * ratio) / 2 - .1), round((320 - height * ratio) / 2 - .1)
            try:
                boxes = decode(values, prepared)
            except ValueError as error:
                scores = values["scores"]
                report["frames"].append({"frame": index, "output_error": str(error),
                                         "score_min": float(scores.min()), "score_max": float(scores.max()),
                                         "score_unique": int(np.unique(scores).size),
                                         "scores_outside_0_1": int(((scores < 0) | (scores > 1)).sum())})
                continue
            result = agreement(expected, boxes)
            report["frames"].append({"frame": index, "agreement": result, "detections": len(boxes)})
            import cv2
            overlay(cv2.imread(str(source)), boxes, run_dir / f"npu_{index}.png")
        if any("output_error" in r for r in report["frames"]):
            report["status"] = "NPU_OUTPUT_INVALID"
        else:
            report["status"] = "NPU_PARITY_PASS" if all(r["agreement"]["gate"] == "PASS" for r in report["frames"]) else "NPU_PARITY_FAIL"
        report["npu_run_duration"] = "includes process startup; run a streaming pipeline benchmark separately for FPS"
    write_json(run_dir / "report.json", report)
    if report["status"] != "NPU_PARITY_PASS" and marker is None:
        raise RuntimeError("Edge QCS6490 NPU gate failed; see " + str(run_dir / "report.json"))
    print(json.dumps({"status": report["status"], "report": str(run_dir / "report.json"),
                      "diagnostic_only": marker is not None, "production_ready": False,
                      "npu_wall_ms_including_process_startup": report["npu_wall_ms_including_process_startup"]}))
    return 0 if report["status"] == "NPU_PARITY_PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("cloud")
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--sdk", type=Path, default=SDK_CLOUD)
    p.add_argument("--output-dir", required=True, type=Path)
    p.set_defaults(run=cloud)
    p = sub.add_parser("prepare-diagnostic", help="prepare existing failed CPU candidate; never quantize again")
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--existing-run", required=True, type=Path)
    p.add_argument("--sdk", type=Path, default=SDK_CLOUD)
    p.add_argument("--output-dir", required=True, type=Path)
    p.set_defaults(run=prepare_diagnostic)
    p = sub.add_parser("edge")
    p.add_argument("--packet", required=True, type=Path)
    p.add_argument("--sdk", type=Path, default=SDK_EDGE)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--diagnostic", action="store_true", help="explicitly allow diagnostic-only CPU-failed packet")
    p.set_defaults(run=edge)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
