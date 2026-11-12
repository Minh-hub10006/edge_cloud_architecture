"""CPU-only diagnostic of RTMDet DLC quantization, with matched-input ONNX replay.

Run the packaged copy on Cloud249. This experiment never admits a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

FLOAT_SHA = "02e71da3de4098f14a48fd1afd9631cba1796aa0898634a5577ce972f7c9448b"
ONNX_SHA = "1078a2c8b62e8ae6ae37f460e7613fa40295b71ab08c11404bd1cd3c695bd4a4"
WRAPPER_SHA = "cc8435098be6aa4f0aa7885e1439796403b27ee383155f8d99977aac050837ed"
GRAPH = "rtmdet_small_fp32"
FINAL_SHAPES = {"boxes": (1, 8400, 4), "scores": (1, 8400, 80)}
STEM_SHAPES = {}
for index, channels in enumerate((16, 16, 32)):
    for suffix in ("conv/Conv_output_0", "activate/Sigmoid_output_0", "activate/Mul_output_0"):
        STEM_SHAPES[f"/backbone/stem/stem.{index}/{suffix}"] = (1, channels, 320, 320)
HEAD_SHAPES = {f"/Transpose_{index}_output_0": (1, 80, side, side)
               for index, side in ((1, 80), (3, 40), (5, 20))}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def metrics(reference, actual):
    if reference.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("tensor shape/nonfinite mismatch")
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    delta = np.abs(a - b)
    return {"mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
            "p95_abs": float(np.percentile(delta, 95)),
            "rel_mae": float(delta.mean() / max(np.abs(a).mean(), 1e-12)),
            "reference_min": float(a.min()), "reference_max": float(a.max()),
            "actual_min": float(b.min()), "actual_max": float(b.max())}


def encodings(info):
    pattern = (rf"(?P<name>\S+) encoding\s*:\s*bitwidth\s+(?P<bits>\d+),\s*"
               rf"min\s+(?P<min>{NUMBER}),\s*max\s+(?P<max>{NUMBER}),\s*"
               rf"scale\s+(?P<scale>{NUMBER}),\s*offset\s+(?P<offset>{NUMBER})")
    result = {}
    for match in re.finditer(pattern, info):
        value = {key: float(match[key]) for key in ("min", "max", "scale", "offset")}
        value["bits"] = int(match["bits"])
        if value["scale"] <= 0 or value["bits"] not in (8, 16, 32):
            raise ValueError("unsupported encoding")
        result[match["name"]] = value
    # DLC info's input summary has no '<name> encoding :' prefix.
    input_pattern = (rf"\|\s*input\s*\|[^\n]*?bitwidth\s+(\d+),\s*min\s+({NUMBER}),\s*"
                     rf"max\s+({NUMBER}),\s*scale\s+({NUMBER}),\s*offset\s+({NUMBER})")
    match = re.search(input_pattern, info)
    if match:
        result["input"] = dict(zip(("bits", "min", "max", "scale", "offset"),
                                   (int(match[1]), *map(float, match.groups()[1:]))))
    return result


def qdq(value, encoding):
    scale, offset, bits = encoding["scale"], encoding["offset"], encoding["bits"]
    if scale <= 0 or not 1 <= bits <= 32:
        raise ValueError("invalid QDQ encoding")
    quant = np.clip(np.rint(value.astype(np.float64) / scale - offset), 0, 2**bits - 1)
    return ((quant + offset) * scale).astype(np.float32)


def read_raw(folder, name, shape):
    path = folder / (name.lstrip("/") + ".raw")
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f"wrong Float32 byte count: {path}")
    value = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite output: {path}")
    return value


def trace_layout(info, name, nchw):
    match = re.search(re.escape(name) + r" \(data type: [^;]+; tensor dimension: \[([^]]+)\]", info)
    if not match:
        raise ValueError(f"tensor dimensions absent from DLC info: {name}")
    dimensions = tuple(map(int, match[1].split(",")))
    nhwc = (nchw[0], nchw[2], nchw[3], nchw[1])
    if dimensions == nhwc and dimensions != nchw:
        return dimensions, True
    if dimensions == nchw and dimensions != nhwc:
        return dimensions, False
    raise ValueError(f"ambiguous/unexpected tensor layout: {name} {dimensions}")


def run(command, log, env, commands, timeout=240):
    record = {"argv": list(map(str, command)), "log": str(log), "timeout_s": timeout}
    commands.append(record)
    started = time.monotonic()
    try:
        with log.open("x") as stream:
            result = subprocess.run(record["argv"], stdout=stream, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout, check=False)
        record["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"command exit {result.returncode}; inspect {log}")
    finally:
        record["elapsed_s"] = time.monotonic() - started


def execute(dlc, inputs, out, tensors, tools, env, commands):
    run([tools["snpe-net-run"], "--container", dlc, "--input_list", inputs,
         "--output_dir", out, "--set_output_tensors", GRAPH + " " + ",".join(tensors)],
        out.with_suffix(".log"), env, commands)


def read_finals(out, count=3):
    return [{name: read_raw(out / f"Result_{i}", name, shape)
             for name, shape in FINAL_SHAPES.items()} for i in range(count)]


def final_metrics(reference, actual):
    active = reference["scores"][0].max(axis=1) >= .1
    result = {name: metrics(reference[name], actual[name]) for name in FINAL_SHAPES}
    result["active_queries"] = int(active.sum())
    if active.any():
        result["active_boxes"] = metrics(reference["boxes"][0, active], actual["boxes"][0, active])
    return result


def instrumentation_pass(reference, actual):
    return (np.allclose(reference["boxes"], actual["boxes"], rtol=1e-5, atol=1e-3)
            and np.allclose(reference["scores"], actual["scores"], rtol=1e-5, atol=1e-5))


def replay_stem(model, trace, input_value, quant_encodings):
    """Hold each node's observed inputs fixed; compare its output to original ONNX.

    Float and QDQ variants are explanatory bounds, not an SNPE emulator. Changed
    weights, fusion, hidden precision and rounding still require separate tests.
    """
    import onnx
    import onnxruntime as ort
    initializers = {item.name: item for item in model.graph.initializer}
    nodes = {output: node for node in model.graph.node for output in node.output}
    observed = {"input": input_value, **trace}
    result = []
    for output in STEM_SHAPES:
        node = nodes[output]
        names = [name for name in node.input if name and name not in initializers]
        values = {name: observed[name] for name in names}
        graph = onnx.helper.make_graph([deepcopy(node)], "isolated_original_op",
            [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, value.shape)
             for name, value in values.items()],
            [onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, observed[output].shape)],
            [deepcopy(initializers[name]) for name in node.input if name in initializers])
        isolated = onnx.helper.make_model(graph, opset_imports=deepcopy(model.opset_import))
        isolated.ir_version = model.ir_version
        onnx.checker.check_model(isolated)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(isolated.SerializeToString(), options,
                                       providers=["CPUExecutionProvider"])
        predicted = session.run(None, values)[0]
        row = {"tensor": output, "op": node.op_type,
               "float_op_on_observed_inputs": metrics(predicted, observed[output])}
        if output in quant_encodings:
            row["with_output_qdq"] = metrics(qdq(predicted, quant_encodings[output]), observed[output])
            adjusted = {name: qdq(value, quant_encodings[name]) if name in quant_encodings else value
                        for name, value in values.items()}
            adjusted_prediction = session.run(None, adjusted)[0]
            row["with_input_and_output_qdq"] = metrics(qdq(adjusted_prediction, quant_encodings[output]), observed[output])
        row["best_tested_rel_mae"] = min(value["rel_mae"] for value in row.values()
                                       if isinstance(value, dict) and "rel_mae" in value)
        result.append(row)
    return result


def verify_packet(root):
    manifest = json.loads((root / "PACKET_MANIFEST.json").read_text())
    for name, wanted in manifest.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or digest(path) != wanted:
            raise ValueError(f"packet checksum mismatch: {name}")
    model = root / "reference/model.onnx"
    if digest(model) != ONNX_SHA:
        raise ValueError("wrong reference ONNX")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/home/stu0/cloud_artifacts/rtmdet_dlc243.xdMW1T"))
    parser.add_argument("--sdk-root", type=Path, default=Path("/home/stu0/cloud_sdks/rtmdet_snpe243.atdhop/2.43.0.260128"))
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run packaged experiment on Linux Cloud249")
    import onnx
    import onnxruntime as ort
    root = Path(__file__).resolve().parent
    model_path = verify_packet(root)
    work, sdk = args.work_root.resolve(strict=True), args.sdk_root.resolve(strict=True)
    dlc = work / "rtmdet_small_fp32.dlc"
    if digest(dlc) != FLOAT_SHA:
        raise ValueError("Float DLC checkpoint changed")
    tools = {name: sdk / "bin/x86_64-linux-clang" / name
             for name in ("snpe-net-run", "snpe-dlc-info", "snpe-dlc-quantize")}
    if any(not path.is_file() for path in tools.values()) or digest(tools["snpe-dlc-quantize"]) != WRAPPER_SHA:
        raise ValueError("SDK tools/checkpoint mismatch")
    if shutil.disk_usage(work).free < 1024**3:
        raise ValueError("need at least 1 GiB free for diagnostic evidence")
    out = Path(tempfile.mkdtemp(prefix="quant_origin_", dir=work))
    print(f"RUN_DIR={out}", flush=True)
    env = os.environ.copy()
    env.update(SNPE_ROOT=str(sdk), QAIRT_ROOT=str(sdk), QAIRT_SDK_ROOT=str(sdk), QNN_SDK_ROOT=str(sdk),
               PATH=f"{sdk / 'bin/x86_64-linux-clang'}:{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
               LD_LIBRARY_PATH=f"{sdk / 'lib/x86_64-linux-clang'}:{Path(sys.prefix) / 'lib'}",
               PYTHONPATH=str(sdk / "lib/python"))
    commands = []
    # Preserve the wrapper used by this run so argument translation can be audited
    # without introducing a dump-only CLI mode into the actual experiment.
    with (out / "quantizer_wrapper.txt").open("x", encoding="utf-8") as stream:
        stream.write(tools["snpe-dlc-quantize"].read_text(errors="replace"))
    report = {"schema_version": 1, "status": "RUNNING", "production_ready": False,
              "runtime": "CPU_DIAGNOSTIC_ONLY", "commands": commands, "cases": [],
              "versions": {"python": sys.version, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "tool_hashes": {name: digest(path) for name, path in tools.items()},
              "child_environment": {key: env[key] for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "SNPE_ROOT")},
              "limits": ["one/three-frame calibration is a diagnostic, not a deployment calibration set",
                         "op replay does not prove SNPE internal arithmetic or unchanged parameter values",
                         "encoding mismatch alone does not identify the quantizer's internal collection path"]}
    try:
        model = onnx.load(model_path)
        all_list, one_list = out / "three_frames.txt", out / "one_frame.txt"
        frame_paths, original = [], []
        for number in (120, 600, 1200):
            tensor = np.load(root / f"reference/frame_{number}.npy", allow_pickle=False)
            if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
                raise ValueError("input contract mismatch")
            path = out / f"frame_{number}.raw"
            with path.open("xb") as stream:
                stream.write(np.ascontiguousarray(tensor.transpose(0, 2, 3, 1)).tobytes())
            frame_paths.append(path)
            with np.load(root / f"reference/frame_{number}.npz", allow_pickle=False) as saved:
                original.append({key: saved[key].copy() for key in FINAL_SHAPES})
        all_list.write_text("".join(str(path) + "\n" for path in frame_paths))
        one_list.write_text(str(frame_paths[0]) + "\n")
        report["input_hashes"] = {path.name: digest(path) for path in frame_paths}
        help_path = out / "quantizer_help.txt"
        run(["/bin/bash", tools["snpe-dlc-quantize"], "--help"], help_path, env, commands)
        help_text = help_path.read_text(errors="replace")
        if any(flag not in help_text for flag in ("--no_weight_quantization", "--act_bitwidth")):
            raise ValueError("required quantizer options absent")
        run([tools["snpe-dlc-info"], "-i", dlc], out / "float_info.txt", env, commands)
        execute(dlc, all_list, out / "float_regular", list(FINAL_SHAPES), tools, env, commands)
        baseline = read_finals(out / "float_regular")
        report["float_vs_onnx"] = [final_metrics(a, b) for a, b in zip(original, baseline)]
        # Observed existing Float DLC errors are well below these smoke bounds.
        if any(item["scores"]["max_abs"] > 1e-3 or item["boxes"]["mean_abs"] > .1
               for item in report["float_vs_onnx"]):
            raise ValueError("Float DLC reference gate failed; stop before quantization")
        trace_names = [*STEM_SHAPES, *HEAD_SHAPES, *FINAL_SHAPES]
        execute(dlc, one_list, out / "float_trace", trace_names, tools, env, commands)
        float_traced_final = read_finals(out / "float_trace", 1)[0]
        report["float_instrumentation_ok"] = instrumentation_pass(baseline[0], float_traced_final)
        if not report["float_instrumentation_ok"]:
            raise ValueError("requesting intermediate outputs changed Float DLC final outputs")
        float_info = (out / "float_info.txt").read_text(errors="replace")
        float_trace = {}
        for name, shape in STEM_SHAPES.items():
            dims, transpose = trace_layout(float_info, name, shape)
            value = read_raw(out / "float_trace/Result_0", name, dims)
            float_trace[name] = value.transpose(0, 3, 1, 2) if transpose else value
        input_value = np.load(root / "reference/frame_120.npy", allow_pickle=False)
        report["float_op_replay"] = replay_stem(model, float_trace, input_value, {})
        if any(row["best_tested_rel_mae"] > .002 for row in report["float_op_replay"]):
            raise ValueError("Float DLC isolated-op replay mismatch; do not interpret quantized replay")
        for name, calibration in (("one_frame_a16_float_weights", one_list),
                                  ("three_frames_a16_float_weights", all_list)):
            print(f"CASE={name}", flush=True)
            case_dir = out / name
            case_dir.mkdir()
            candidate = case_dir / "candidate.dlc"
            row = {"name": name, "status": "RUNNING"}
            report["cases"].append(row)
            flags = ["--no_weight_quantization", "--act_bitwidth", "16"]
            run(["/bin/bash", tools["snpe-dlc-quantize"], "--input_dlc", dlc, "--input_list", calibration,
                 "--output_dlc", candidate, *flags], case_dir / "quantize.log", env, commands, timeout=600)
            row["artifact_sha256"] = digest(candidate)
            run([tools["snpe-dlc-info"], "-i", candidate], case_dir / "info.txt", env, commands)
            info = (case_dir / "info.txt").read_text(errors="replace")
            encoded = encodings(info)
            row["encodings"] = {key: encoded.get(key) for key in ("input", *STEM_SHAPES, *HEAD_SHAPES)}
            if any(key not in encoded or encoded[key]["bits"] != 16 for key in ("input", *STEM_SHAPES)):
                raise ValueError("actual A16 encoding gate failed; see info, argv and saved SDK wrapper")
            execute(candidate, all_list, case_dir / "regular", list(FINAL_SHAPES), tools, env, commands)
            actual = read_finals(case_dir / "regular")
            row["final_vs_float"] = [final_metrics(a, b) for a, b in zip(baseline, actual)]
            execute(candidate, one_list, case_dir / "trace", trace_names, tools, env, commands)
            traced_final = read_finals(case_dir / "trace", 1)[0]
            row["instrumentation_ok"] = instrumentation_pass(actual[0], traced_final)
            row["instrumentation_difference"] = final_metrics(actual[0], traced_final)
            if not row["instrumentation_ok"]:
                row["status"] = "INSTRUMENTATION_CHANGED_OUTPUT_STOP_REPLAY"
                continue
            trace = {}
            for tensor, shape in STEM_SHAPES.items():
                dims, transpose = trace_layout(info, tensor, shape)
                value = read_raw(case_dir / "trace/Result_0", tensor, dims)
                trace[tensor] = value.transpose(0, 3, 1, 2) if transpose else value
            row["cumulative_stem_errors"] = {key: metrics(float_trace[key], trace[key]) for key in STEM_SHAPES}
            row["op_replay"] = replay_stem(model, trace, input_value, encoded)
            row["encoding_coverage_on_calibration_frame"] = {}
            for tensor, shape in {**STEM_SHAPES, **HEAD_SHAPES}.items():
                if tensor not in encoded:
                    continue
                value = (float_trace[tensor] if tensor in float_trace else
                         read_raw(out / "float_trace/Result_0", tensor, shape))
                enc = encoded[tensor]
                outside = (value < enc["min"] - 2 * enc["scale"]) | (value > enc["max"] + 2 * enc["scale"])
                row["encoding_coverage_on_calibration_frame"][tensor] = {
                    "float_min": float(value.min()), "float_max": float(value.max()),
                    "outside_by_more_than_two_steps_fraction": float(outside.mean()),
                    "encoding": enc}
            row["first_replay_review_tensor"] = next((x["tensor"] for x in row["op_replay"]
                                                       if x["best_tested_rel_mae"] > .01), None)
            row["status"] = "DIAGNOSTIC_COMPLETE_NOT_ADMITTED"
            print(json.dumps({"case": name, "frame120_max_score": float(actual[0]["scores"].max()),
                              "frame120_box_mae": row["final_vs_float"][0]["boxes"]["mean_abs"],
                              "first_replay_review_tensor": row["first_replay_review_tensor"]}), flush=True)
        report["status"] = "DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED"
    except Exception as error:
        report["status"] = "STOPPED"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        report["source_float_unchanged"] = digest(dlc) == FLOAT_SHA
        write_json(out / "report.json", report)
        # Keep raw traces/candidates on249; portable review contains metrics/logs/encodings only.
        review = out / "review.zip"
        with zipfile.ZipFile(review, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".txt", ".log") and path.stat().st_size < 20_000_000:
                    archive.write(path, path.relative_to(out).as_posix())
        print(json.dumps({"report": str(out / "report.json"), "review_zip": str(review),
                          "review_sha256": digest(review), "status": report["status"]}), flush=True)
    return 0 if report["status"] == "DIAGNOSTIC_COMPLETE_REVIEW_REQUIRED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
