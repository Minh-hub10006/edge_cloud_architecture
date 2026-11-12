"""Isolated CPU -> QCS6490 quantization sweep. Never changes production artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import Box
from scripts.check_rtmdet_snpe_outputs import SHAPES, decode, digest, matching, native_output, raw_metrics
from scripts.rebuild_rtmdet_small import CHECKPOINT_SHA256, REVISION, compare_detections

FLOAT_SHA = "02e71da3de4098f14a48fd1afd9631cba1796aa0898634a5577ce972f7c9448b"
OLD_QUANT_SHA = "343b515f0b7861870c1eb32a91f9d48cf7c6ad7bf000e7ebe1a55f5b5c773667"
ONNX_SHA = "1078a2c8b62e8ae6ae37f460e7613fa40295b71ab08c11404bd1cd3c695bd4a4"
QUANTIZER_SHA = "cc8435098be6aa4f0aa7885e1439796403b27ee383155f8d99977aac050837ed"
GRAPH = "rtmdet_small_fp32"
EDGE_SDK = "/home/ubuntu/sigm/qairt/2.43.0.260128"
CASES = (
    ("keep_weights_float_control", ["--no_weight_quantization", "--act_bitwidth", "8"], False),
    ("int8_per_channel", ["--weights_bitwidth", "8", "--act_bitwidth", "8", "--use_per_channel_quantization"], True),
    ("int8_per_channel_enhanced", ["--weights_bitwidth", "8", "--act_bitwidth", "8", "--use_per_channel_quantization",
                                   "--param_quantizer", "enhanced", "--act_quantizer", "enhanced"], True),
)


class AccessBlocked(RuntimeError):
    """Stop the whole sweep at a remote access/approval boundary."""


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_bundle(root):
    manifest = json.loads((root / "SWEEP_BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    required = {"scripts/rtmdet_quant_sweep.py"}
    required.update(f"sweep_references/frame_{number:06d}/{name}" for number in (120, 600, 1200)
                    for name in ("source.png", "reference_input.npy", "reference_outputs.npz", "rebuild_report.json"))
    if not required.issubset(manifest):
        raise ValueError("incomplete bundle manifest")
    for name, wanted in manifest.items():
        if not name or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("unsafe bundle member")
        if digest(root / name) != wanted:
            raise ValueError(f"bundle checksum mismatch: {name}")


def load_frames(root):
    frames = []
    for number in (120, 600, 1200):
        folder = root / "sweep_references" / f"frame_{number:06d}"
        report = json.loads((folder / "rebuild_report.json").read_text(encoding="utf-8"))
        if (report["export_gate"] != "PASS" or report["source_revision"] != REVISION
                or report["checkpoint_sha256"] != CHECKPOINT_SHA256 or report["artifact_sha256"] != ONNX_SHA
                or report["frame_index"] != number or digest(folder / "source.png") != report["evidence"]["image_sha256"]):
            raise ValueError("reference source/shape/frame gate failed")
        tensor = np.load(folder / "reference_input.npy", allow_pickle=False)
        if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError("reference tensor gate failed")
        with np.load(folder / "reference_outputs.npz", allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in SHAPES}
        for name, shape in SHAPES.items():
            if values[name].shape != shape or not np.isfinite(values[name]).all():
                raise ValueError("reference output shape/finite gate failed")
        boxes = decode(values, report)
        compare_detections([Box(**box) for box in report["boxes"]], boxes)
        frames.append({"id": f"frame_{number:06d}", "folder": folder, "report": report,
                       "input": np.ascontiguousarray(tensor.transpose(0, 2, 3, 1), dtype="<f4"),
                       "values": values, "boxes": boxes})
    return frames


def agreement(reference, actual):
    result = matching(reference, actual)
    pairs = result.pop("pairs")
    matched = len(pairs)
    result.update(reference_coverage=matched / max(1, len(reference)),
                  candidate_agreement=matched / max(1, len(actual)),
                  median_iou=float(np.median([p["iou"] for p in pairs])) if pairs else 0.,
                  matched_score_error_p95=float(np.percentile([
                      abs(reference[p["reference_index"]].confidence - actual[p["native_index"]].confidence)
                      for p in pairs], 95)) if pairs else None)
    return result


def smoke_pass(metrics):
    return (metrics["reference_coverage"] >= .9 and metrics["candidate_agreement"] >= .9
            and metrics["median_iou"] >= .8 and metrics["matched_score_error_p95"] is not None
            and metrics["matched_score_error_p95"] <= .12)


def inspect_outputs(folder, frames, *, npu=False):
    outcomes = []
    for index, frame in enumerate(frames):
        raw = folder / frame["id"] / "Result_0" if npu else folder / f"Result_{index}"
        values = {name: native_output(raw / f"{name}.raw", shape) for name, shape in SHAPES.items()}
        boxes = decode(values, frame["report"])
        metrics = agreement(frame["boxes"], boxes)
        try:
            strict = compare_detections(frame["boxes"], boxes)
        except ValueError as error:
            strict = {"gate": "FAIL", "reason": str(error)}
        outcomes.append({"frame_id": frame["id"], "reference_count": len(frame["boxes"]),
                         "candidate_count": len(boxes), "agreement": metrics,
                         "smoke_gate": "PASS" if smoke_pass(metrics) else "FAIL", "strict_fp32_parity": strict,
                         "raw_metrics": raw_metrics(frame["values"], values),
                         "boxes": [asdict(box) for box in boxes],
                         "output_hashes": {name: digest(raw / f"{name}.raw") for name in SHAPES}})
    return outcomes


def outputs_pass(outcomes):
    return len(outcomes) == 3 and all(item["smoke_gate"] == "PASS" for item in outcomes)


def prepare_output_arg(help_text):
    if "--set_output_tensors" not in help_text:
        raise ValueError("SDK graph prepare lacks explicit output selection; stop")
    # Runtime CLI and graph-prepare CLI can expose different argument grammar.
    section = help_text[help_text.index("--set_output_tensors"):]
    section = re.split(r"\n\s*(?:\[\s*)?--[a-zA-Z_]", section, maxsplit=1)[0]
    if re.search(r"graphA\s+tensor|graph name.*tensor|Multi Graph", section, re.I):
        return f"{GRAPH} boxes,scores"
    if re.search(r"comma[- ]separated|comma separated", section, re.I):
        return "boxes,scores"
    raise ValueError("unknown graph-prepare output syntax; see help log, no guessing")


def select_winner(cases):
    valid = [case for case in cases if case["deployable_int8"] and case.get("npu_gate") == "PASS"]
    if not valid:
        return None
    def rank(case):
        frames = case["npu_results"]
        return (min(min(f["agreement"]["reference_coverage"], f["agreement"]["candidate_agreement"]) for f in frames),
                min(f["agreement"]["median_iou"] for f in frames),
                -max(f["agreement"]["matched_score_error_p95"] for f in frames))
    return max(valid, key=rank)["name"]


def run(cmd, log, env, *, timeout=900):
    with Path(log).open("xb") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False)
    if result.returncode:
        if Path(cmd[0]).name in ("ssh", "scp"):
            detail = Path(log).read_text(errors="replace")
            if result.returncode == 255 or re.search(r"Permission denied|Host key verification failed|REMOTE HOST IDENTIFICATION", detail, re.I):
                raise AccessBlocked(f"remote access failed; see {log}")
        raise RuntimeError(f"command exit {result.returncode}; see {log}")


def edge_commands(edge_case, frames):
    """Private SDK env, new case directory, checksum-bound inputs and named outputs."""
    q = shlex.quote
    edge_runner = f"{EDGE_SDK}/bin/aarch64-oe-linux-gcc11.2/snpe-net-run"
    script = ("set -euo pipefail\n" + f"cd {q(edge_case)}\nsha256sum -c TRANSFER_SHA256SUMS\n"
        + f"export LD_LIBRARY_PATH={q(EDGE_SDK + '/lib/aarch64-oe-linux-gcc11.2')}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n"
        + f"export ADSP_LIBRARY_PATH={q(EDGE_SDK + '/lib/hexagon-v68/unsigned;' + EDGE_SDK + '/lib/aarch64-oe-linux-gcc11.2;/usr/lib/rfsa/adsp;/dsp;/opt')}\n")
    for frame in frames:
        tag = frame["id"]
        script += (f"printf '%s\\n' {q(edge_case + '/inputs/' + tag + '.raw')} > {q(tag + '.txt')}\n"
            + f"{q(edge_runner)} --container candidate_qcs6490.dlc --input_list {q(tag + '.txt')} "
            + f"--output_dir {q('npu/' + tag)} --set_output_tensors {q(GRAPH + ' boxes,scores')} --use_dsp "
            + f"--profiling_level basic > {q('npu_' + tag + '.log')} 2>&1\n"
            + f"cp {q('npu_' + tag + '.log')} {q('npu/' + tag + '/run.log')}\n")
    return script + "sha256sum -c TRANSFER_SHA256SUMS\n"


def checkpoint_unchanged(original):
    try:
        return all(digest(path) == wanted for path, wanted in original.items())
    except OSError:
        return False


def pack_review(output):
    review = output / "review.zip"
    with zipfile.ZipFile(review, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".log", ".txt", ".png", ".raw"):
                archive.write(path, path.relative_to(output).as_posix())
    return review


def overlays(folder, frames, outcomes, label):
    import cv2
    cv2.setNumThreads(1)
    for frame, outcome in zip(frames, outcomes):
        image = cv2.imread(str(frame["folder"] / "source.png"))
        if image is None:
            raise ValueError("overlay image missing")
        canvases = []
        for title, boxes, color in (("FP32", frame["boxes"], (0, 210, 0)),
                                    (label, [Box(**box) for box in outcome["boxes"]], (0, 165, 255))):
            canvas = image.copy()
            h, w = canvas.shape[:2]
            for box in boxes:
                p1, p2 = (round(box.left * w), round(box.top * h)), (round(box.right * w), round(box.bottom * h))
                cv2.rectangle(canvas, p1, p2, color, 2)
                cv2.putText(canvas, f"{box.class_name} {box.confidence:.2f}", (p1[0], max(15, p1[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            cv2.rectangle(canvas, (0, 0), (w, 65), (25, 25, 25), -1)
            cv2.putText(canvas, f"{title}: {len(boxes)} | conf=0.1", (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2)
            canvases.append(cv2.resize(canvas, (960, 540)))
        if not cv2.imwrite(str(folder / f"{frame['id']}_comparison.png"), np.concatenate(canvases, axis=1)):
            raise OSError("overlay write failed")


def check_calibration(work):
    folder = work / "calibration_nhwc100"
    report = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
    if (report["reference_artifact_sha256"] != ONNX_SHA or report["input_layout"] != "nhwc"
            or report["shape"] != [1, 640, 640, 3] or report["dtype"] != "float32"
            or report["byte_order"] != "little" or report["sample_count"] != 100
            or len(report["samples"]) != 100):
        raise ValueError("calibration contract mismatch")
    paths = []
    for item in report["samples"]:
        name = item["file"]
        if Path(name).name != name or "\\" in name or name in (".", ".."):
            raise ValueError("unsafe calibration filename")
        path = folder / name
        values = native_output(path, (1, 640, 640, 3))
        if (digest(path) != item["tensor_sha256"]
                or hashlib.sha256(np.ascontiguousarray(values.transpose(0, 3, 1, 2)).tobytes()).hexdigest()
                != item["source_nchw_sha256"]):
            raise ValueError("calibration hash/layout mismatch")
        paths.append(str(path))
    if (folder / "input_list.txt").read_text().splitlines() != paths:
        raise ValueError("calibration input_list path binding mismatch")
    return folder / "input_list.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/home/stu0/cloud_artifacts/rtmdet_dlc243.xdMW1T"))
    parser.add_argument("--sdk-root", type=Path, default=Path("/home/stu0/cloud_sdks/rtmdet_snpe243.atdhop/2.43.0.260128"))
    parser.add_argument("--edge-target", default="ubuntu@192.168.180.246")
    args = parser.parse_args()
    if sys.platform != "linux" or not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+", args.edge_target):
        parser.error("run on Linux249 with normal SSH user@host; no injected SSH options")
    root, work, sdk = Path(__file__).resolve().parents[1], args.work_root.resolve(strict=True), args.sdk_root.resolve(strict=True)
    if (root / "sweep_review.zip").exists():
        parser.error("this isolated lane already has a review ZIP; use a fresh lane for another run")
    verify_bundle(root)
    frames = load_frames(root)
    calibration = check_calibration(work)
    original = {work / "rtmdet_small_fp32.dlc": FLOAT_SHA, work / "rtmdet_small_int8.dlc": OLD_QUANT_SHA}
    if any(digest(path) != wanted for path, wanted in original.items()):
        raise ValueError("original DLC checkpoint mismatch; no quantization run")
    bin_dir = sdk / "bin/x86_64-linux-clang"
    tools = {name: bin_dir / name for name in ("snpe-net-run", "snpe-dlc-quantize", "snpe-dlc-graph-prepare", "snpe-dlc-info")}
    if not all(path.is_file() and os.access(path, os.X_OK) for path in tools.values()):
        raise ValueError("required SDK tool missing")
    if digest(tools["snpe-dlc-quantize"]) != QUANTIZER_SHA:
        raise ValueError("quantizer checkpoint differs from verified shell wrapper; stop")
    original[tools["snpe-dlc-quantize"]] = QUANTIZER_SHA
    import cv2  # Preflight only; no package install or environment mutation.
    token = "quant_sweep_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = work / token
    output.mkdir()
    env = os.environ.copy()
    env.update(QAIRT_ROOT=str(sdk), SNPE_ROOT=str(sdk),
               PATH=f"{bin_dir}:{Path(sys.executable).parent}:{env.get('PATH', '')}",
               LD_LIBRARY_PATH=f"{sdk / 'lib/x86_64-linux-clang'}:{Path(sys.prefix) / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
               PYTHONPATH=f"{sdk / 'lib/python'}:{env.get('PYTHONPATH', '')}")
    summary = {"schema_version": 1, "run_dir": str(output), "frames": [f["id"] for f in frames],
               "quality_status": "NOT_EVALUATED", "production_ready": False, "cases": [],
               "note": "CPU default is float computation; gates are three-frame FP32 agreement, NOT mAP or admission.",
               "smoke_criteria": {"same_class_iou": .5, "min_reference_coverage": .9,
                                  "min_candidate_agreement": .9, "min_median_iou": .8, "max_score_error_p95": .12},
               "winner": None}
    summary["checkpoint_hashes"] = {str(path): value for path, value in original.items()}
    control_dir = tempfile.TemporaryDirectory(prefix="rtms_")
    ssh_base = ["-o", "ControlMaster=auto", "-o", "ControlPersist=600", "-o", f"ControlPath={control_dir.name}/c",
                "-o", "ConnectTimeout=10"]
    ssh = ["ssh", *ssh_base, "-o", "BatchMode=yes", args.edge_target]
    scp = ["scp", *ssh_base, "-o", "BatchMode=yes"]
    connected = False
    print(f"RUN_DIR={output}", flush=True)
    try:
        for key in ("snpe-dlc-quantize", "snpe-dlc-graph-prepare"):
            cmd = (["/bin/bash", str(tools[key])] if key == "snpe-dlc-quantize" else [str(tools[key])]) + ["--help"]
            run(cmd, output / f"{key}_help.txt", env, timeout=60)
        quant_help = (output / "snpe-dlc-quantize_help.txt").read_text(errors="replace")
        for _, flags, _ in CASES:
            if any(flag not in quant_help for flag in flags if flag.startswith("--")):
                raise ValueError("quantizer option gate failed; see recorded help")
        prep_help = (output / "snpe-dlc-graph-prepare_help.txt").read_text(errors="replace")
        print("SSH preflight246: may ask for your existing SSH password once.", flush=True)
        edge_runner = f"{EDGE_SDK}/bin/aarch64-oe-linux-gcc11.2/snpe-net-run"
        preflight = (f"test \"$(uname -m)\" = aarch64 && test -x {shlex.quote(edge_runner)} "
                     "&& test -d /home/ubuntu/edge_artifacts && "
                     "df -Pk /home/ubuntu/edge_artifacts | awk 'NR==2 {if ($4<524288) exit 1}'")
        subprocess.run(["ssh", *ssh_base, args.edge_target, preflight], check=True, timeout=90)
        connected = True
        inputs = output / "inputs"
        inputs.mkdir()
        for frame in frames:
            with (inputs / f"{frame['id']}.raw").open("xb") as stream:
                stream.write(frame["input"].tobytes())
        input_list = output / "input_list.txt"
        with input_list.open("x") as stream:
            stream.write("".join(str(inputs / f"{f['id']}.raw") + "\n" for f in frames))
        runner_outputs = f"{GRAPH} boxes,scores"
        baseline = output / "float_baseline"
        print("Float baseline: CPU, three frames", flush=True)
        run([str(tools["snpe-net-run"]), "--container", str(work / "rtmdet_small_fp32.dlc"),
             "--input_list", str(input_list), "--output_dir", str(baseline),
             "--set_output_tensors", runner_outputs], output / "float_baseline.log", env, timeout=180)
        summary["float_baseline"] = inspect_outputs(baseline, frames)
        if not all(f["strict_fp32_parity"]["gate"] == "PASS" for f in summary["float_baseline"]):
            raise ValueError("float baseline differs from oracle on at least one frame; stop before tuning")
        for name, flags, deployable in CASES:
            case_dir = output / name
            case_dir.mkdir()
            case = {"name": name, "deployable_int8": deployable, "cpu_gate": "NOT_RUN", "npu_gate": "NOT_RUN"}
            summary["cases"].append(case)
            print(f"=== {name}: quantize -> CPU check ===", flush=True)
            try:
                dlc = case_dir / "candidate.dlc"
                run(["/bin/bash", str(tools["snpe-dlc-quantize"]), "--input_dlc", str(work / "rtmdet_small_fp32.dlc"),
                     "--input_list", str(calibration), "--output_dlc", str(dlc), *flags], case_dir / "quantize.log", env)
                case["quantized_sha256"] = digest(dlc)
                run([str(tools["snpe-dlc-info"]), "-i", str(dlc)], case_dir / "dlc_info.txt", env, timeout=60)
                run([str(tools["snpe-net-run"]), "--container", str(dlc), "--input_list", str(input_list),
                     "--output_dir", str(case_dir / "cpu"), "--set_output_tensors", runner_outputs],
                    case_dir / "cpu.log", env, timeout=180)
                case["cpu_results"] = inspect_outputs(case_dir / "cpu", frames)
                case["cpu_gate"] = "PASS" if outputs_pass(case["cpu_results"]) else "FAIL"
                overlays(case_dir, frames, case["cpu_results"], name + " CPU")
                print(f"CPU_GATE={case['cpu_gate']}", flush=True)
                if not deployable or case["cpu_gate"] != "PASS":
                    continue
                # Cache CLI readiness must not prevent the three CPU diagnostic cases.
                prep_outputs = prepare_output_arg(prep_help)
                if "--htp_socs" not in prep_help:
                    raise ValueError("SDK lacks explicit QCS6490 target selection")
                cached = case_dir / "candidate_qcs6490.dlc"
                run([str(tools["snpe-dlc-graph-prepare"]), "--input_dlc", str(dlc), "--output_dlc", str(cached),
                     "--htp_socs", "qcs6490", "--set_output_tensors", prep_outputs], case_dir / "prepare.log", env)
                run([str(tools["snpe-dlc-info"]), "-i", str(cached)], case_dir / "cached_info.txt", env, timeout=60)
                info = (case_dir / "cached_info.txt").read_text(errors="replace").split("Cache Info:")[-1]
                if not all(mark in info for mark in ("HTP_V68_QCS6490_2MB", "boxes", "scores")):
                    raise ValueError("cache target/dual-output gate failed; do not upload to246")
                edge_case = f"/home/ubuntu/edge_artifacts/{token}/{name}"
                run([*ssh, f"mkdir -p {shlex.quote(str(Path(edge_case).parent))} && mkdir {shlex.quote(edge_case)}"],
                    case_dir / "ssh_mkdir.log", os.environ.copy(), timeout=30)
                transfer_hashes = case_dir / "TRANSFER_SHA256SUMS"
                with transfer_hashes.open("x") as stream:
                    stream.write(f"{digest(cached)}  candidate_qcs6490.dlc\n")
                    stream.write("".join(f"{digest(inputs / (f['id'] + '.raw'))}  inputs/{f['id']}.raw\n" for f in frames))
                run([*scp, "-r", str(cached), str(inputs), str(transfer_hashes), f"{args.edge_target}:{edge_case}/"],
                    case_dir / "upload.log", os.environ.copy(), timeout=180)
                edge_script = edge_commands(edge_case, frames)
                print(f"NPU246 test: {name}, three frames", flush=True)
                run([*ssh, "bash -c " + shlex.quote(edge_script)], case_dir / "edge_run.log", os.environ.copy(), timeout=240)
                run([*scp, "-r", f"{args.edge_target}:{edge_case}/npu", str(case_dir) + "/"],
                    case_dir / "download.log", os.environ.copy(), timeout=180)
                case["npu_results"] = inspect_outputs(case_dir / "npu", frames, npu=True)
                case["npu_gate"] = "PASS" if outputs_pass(case["npu_results"]) else "FAIL"
                case["edge_candidate_dir"] = edge_case
                case["cached_sha256"] = digest(cached)
                (case_dir / "npu_overlays").mkdir()
                overlays(case_dir / "npu_overlays", frames, case["npu_results"], name + " NPU")
                print(f"NPU_GATE={case['npu_gate']}", flush=True)
            except AccessBlocked:
                raise
            except Exception as error:
                case["error"] = f"{type(error).__name__}: {error}"
                print(case["error"], flush=True)
        summary["winner"] = select_winner(summary["cases"])
        summary["status"] = "SMOKE_CANDIDATE_FOUND" if summary["winner"] else "NO_ADMITTED_CANDIDATE"
    except Exception as error:
        summary["fatal_error"] = f"{type(error).__name__}: {error}"
        summary["status"] = "STOPPED"
    finally:
        summary["original_artifacts_unchanged"] = checkpoint_unchanged(original)
        if not summary["original_artifacts_unchanged"]:
            summary["status"], summary["winner"] = "CHECKPOINT_CHANGED_STOP", None
        write_json(output / "summary.json", summary)
        review = pack_review(output)
        # Fixed handoff path in the NEW lane only, created exclusively (never overwrites).
        with review.open("rb") as source, (root / "sweep_review.zip").open("xb") as target:
            shutil.copyfileobj(source, target)
        if connected:
            try:
                subprocess.run(["ssh", *ssh_base, "-O", "exit", args.edge_target], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        control_dir.cleanup()
        print(json.dumps({"status": summary["status"], "winner": summary["winner"],
                          "review_zip": str(root / "sweep_review.zip"), "review_sha256": digest(review),
                          "production_ready": False}), flush=True)
    return 0 if summary["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
