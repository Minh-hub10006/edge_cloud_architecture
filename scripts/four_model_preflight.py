"""Read-only G0 inventory. Emits JSON; never starts models or changes services."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inventory(path, expected=None):
    path = Path(path)
    result = {"path": str(path), "expected_sha256": expected}
    try:
        result.update(resolved=str(path.resolve(strict=True)), bytes=path.stat().st_size,
                      sha256=digest(path))
        result["gate"] = "MATCH" if expected == result["sha256"] else (
            "MISMATCH" if expected else "UNPINNED")
    except (OSError, ValueError) as exc:
        result.update(gate="UNAVAILABLE", error=type(exc).__name__)
    return result


def command(argv):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"return_code": done.returncode, "stdout": done.stdout[:12000],
                "stderr": done.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def source_inventory(root):
    root = Path(root).resolve()
    result = {"root": str(root), "gate": "UNKNOWN", "files": {}}
    if not (root / "inference/catalog.py").is_file():
        result["reason"] = "Explicit source root is not a platform release"
        return result
    # Source only, bounded directories; no .env, credentials, model/video or run data.
    allowed = {".py", ".js", ".cjs", ".sh", ".md", ".proto", ".toml"}
    dirs = ("inference", "contracts", "edge_agent", "cloud_core", "web", "media",
            "benchmark", "scripts", "tests", "docs")
    files = [root / name for name in ("README.md", "IMPLEMENTATION_PLAN.md", "pyproject.toml")]
    for directory in dirs:
        files.extend((root / directory).rglob("*"))
    for path in sorted(set(files)):
        if path.is_symlink() or not path.is_file() or path.suffix not in allowed:
            continue
        relative = path.relative_to(root)
        if any(p.startswith(".") or p in {"__pycache__", "results", "artifacts", "node_modules"}
               for p in relative.parts):
            continue
        if path.stat().st_size > 4 * 1024 * 1024:
            continue
        result["files"][relative.as_posix()] = digest(path)
    canonical = json.dumps(result["files"], sort_keys=True, separators=(",", ":")).encode()
    result["inventory_sha256"] = hashlib.sha256(canonical).hexdigest()
    result["gate"] = "RECORDED_NOT_ACCEPTED"
    return result


def source_candidates(parent):
    parent = Path(parent)
    candidates = []
    if not parent.is_dir():
        return candidates
    for directory in [parent, *sorted(p for p in parent.iterdir() if p.is_dir())]:
        for item in (directory, directory / "source"):
            if (item / "inference/catalog.py").is_file():
                candidates.append(str(item))
    return sorted(set(candidates))


def processes(parent):
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    parent = Path(parent).resolve()
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            cwd = (item / "cwd").resolve(strict=True)
            cwd.relative_to(parent)
            found.append({"pid": int(item.name), "cwd": str(cwd),
                          "executable": str((item / "exe").resolve(strict=True))})
        except (OSError, ValueError):
            pass
    return found  # No command arguments or process environments/secrets collected.


def collect(node, source=None):
    home = Path("/home/ubuntu" if node == "edge246" else "/home/stu0")
    parent = home / "edge_video_platform_benchmark_overlap_20261004"
    if node == "edge246":
        artifacts = [
            (home / "edge_artifacts/npu243/yolov8n_320_int8_split_traffic100.dlc", None),
            (parent / "yolo11s320_split_diag.NHBFPo/npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc",
             "1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3"),
        ]
        sdk = home / "sigm/qairt/2.43.0.260128"
        tools = [sdk / "bin/aarch64-oe-linux-gcc11.2/snpe-net-run"]
        libraries = [sdk / "lib/aarch64-oe-linux-gcc11.2/libSNPE.so"]
    else:
        artifacts = [
            (home / "cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx",
             "265508cc3dc296cee5fd21b7ff818196358a234eebdce5e6e0a9949b5998fca9"),
            (home / "cloud_artifacts/rtdetrv2_s_candidate_20261005/rtdetrv2_s.onnx.data",
             "fee52913900260b0af43d97b28fdfce10dfabe0106ffe86a372c242bec7f5202"),
            (home / "cloud_artifacts/lwdetr_large_candidate_20261006/lwdetr_large_coco.onnx",
             "4f2bb0afdd7f1a1e7ba95406b02106bcc9aefb75e2d1280293c3cbcbbfa07bff"),
        ]
        sdk = home / "cloud_sdks/rtmdet_snpe243.atdhop/2.43.0.260128"
        tools = [sdk / "bin/x86_64-linux-clang/snpe-dlc-quant"]
        libraries = [home / "miniconda3/envs/qnn/lib/libc++.so.1"]
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "onnx", "onnxruntime",
                 "onnxruntime-gpu", "grpcio", "protobuf", "uvicorn", "av"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    providers = command([sys.executable, "-B", "-c",
                         "import onnxruntime as o; print(o.get_available_providers())"])
    disk = shutil.disk_usage(home if home.exists() else Path.cwd())
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "node": node, "host": platform.node(), "machine": platform.machine(),
        "python": sys.executable, "python_version": platform.python_version(),
        "status": "G0_EVIDENCE_COLLECTED_REVIEW_REQUIRED", "baseline_gate": "NOT_RUN",
        "quality_gate": "NOT_EVALUATED", "no_service_changes": True,
        "source": source_inventory(source) if source else {"gate": "UNKNOWN", "root": None},
        "source_candidates": source_candidates(parent), "process_locations": processes(parent),
        "artifacts": [inventory(path, expected) for path, expected in artifacts],
        "sdk": str(sdk), "runtime_files": [inventory(path) for path in tools + libraries],
        "library_dependencies": {str(path): command(["ldd", str(path)]) for path in tools + libraries
                                 if path.is_file()},
        "packages": packages, "ort_available_providers_not_inference_proof": providers,
        "disk_free_bytes": disk.free, "memory": command(["free", "-m"]),
        "listening_ports": command(["ss", "-ltn"]),
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader"]) if node == "cloud249" else None,
        "rollback": {"status": "REQUIRES_REVIEW", "note":
            "Retain current source/artifacts. Record confirmed launch config and baseline smoke before G1; "
            "never kill/restart an unidentified PID. This collector does not create a deployment lane."},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, choices=("edge246", "cloud249"))
    parser.add_argument("--source", type=Path, help="confirmed release root; omit if unknown")
    parser.add_argument("--output", type=Path, help="new report file; existing files are never overwritten")
    args = parser.parse_args()
    report = collect(args.node, args.source)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({"report": str(args.output), "status": report["status"],
                          "baseline_gate": report["baseline_gate"],
                          "artifacts": [{"path": a["path"], "gate": a["gate"]}
                                        for a in report["artifacts"]],
                          "source_candidates": report["source_candidates"]}))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
