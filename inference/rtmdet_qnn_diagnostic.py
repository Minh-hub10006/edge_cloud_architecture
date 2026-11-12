"""Hash-pinned, standalone QNN real-image diagnostic; NOT a pipeline adapter.

Quantization is observed from the supplied EPContext wrapper. Image/head semantics
are an explicit unverified MMDetection hypothesis until source/parity is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import subprocess
import numpy as np

from .model_artifact import sha256
from .detection_decode import rtmdet_decode

MODEL_ID = "rtmdet_s_qnn_candidate_846a03fd2674"
BUNDLE_HASHES = {
    "model.onnx": "6ff8ee8fcdd612b5fbc4e5cf7a7c7df3de3ded80a3a738a486cd088ed7302367",
    "model.bin": "846a03fd26747c152afc1f431ab9a90e1870699e152e13880d420a7fc317a4c3",
}
SDK_VERSION = "2.50.0.260828"
# ONNX QuantizeLinear uses (q - zero_point) * scale. QNN offset is -zero_point.
ENCODINGS = {
    "image": ((1, 3, 640, 640), 0.016547486186027527, 96),
    "output_0": ((1, 80, 80, 80), 0.06468451023101807, 255),
    "output_3": ((1, 4, 80, 80), 0.9631467461585999, 0),
    "output_1": ((1, 80, 40, 40), 0.09449118375778198, 255),
    "output_4": ((1, 4, 40, 40), 1.2684797048568726, 5),
    "output_2": ((1, 80, 20, 20), 0.07458331435918808, 255),
    "output_5": ((1, 4, 20, 20), 1.5038264989852905, 0),
}
HEADS = (("output_0", "output_3", 8), ("output_1", "output_4", 16),
         ("output_2", "output_5", 32))
REFERENCE_CONFIG = "https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"
REFERENCE_HEAD = "https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py"
REFERENCE_QAI_MODEL = "https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/rtmdet/model.py"
PREPROCESSING_PROFILES = ("mmdet_bgr_normalized", "qai_rgb01")


def preprocessing_contract(profile):
    if profile not in PREPROCESSING_PROFILES:
        raise ValueError(f"unknown diagnostic preprocessing profile: {profile}")
    common = {"preprocessing_profile": profile, "resize": "keep_ratio640; pad right/bottom114",
              "reference_preprocessing_match": "UNVERIFIED"}
    if profile == "qai_rgb01":
        return {**common, "color": "RGB", "host_normalization": "divide pixels by255 only",
                "normalize_mean": None, "normalize_std": None,
                "hypothesis": "context contains its own BGR conversion and mean/std normalization",
                "reference_preprocessing": REFERENCE_QAI_MODEL,
                "reference_variant_warning": "current Qualcomm recipe defaults to M, not proven source of supplied Small context"}
    return {**common, "color": "BGR", "host_normalization": "(pixels-mean)/std",
            "normalize_mean": [103.53, 116.28, 123.675], "normalize_std": [57.375, 57.12, 58.395],
            "hypothesis": "context expects externally BGR-normalized input",
            "reference_preprocessing": REFERENCE_CONFIG}


def verify_bundle(folder: Path):
    folder = Path(folder).expanduser().resolve(strict=True)
    for name, wanted in BUNDLE_HASHES.items():
        path = (folder / name).resolve(strict=True)
        if path.parent != folder or not path.is_file() or sha256(path) != wanted:
            raise ValueError(f"QNN diagnostic requires original supplied bundle: {name} checksum/path mismatch")
    return folder


def quantize(value, name="image"):
    shape, scale, zero_point = ENCODINGS[name]
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid {name} tensor shape/values")
    return np.ascontiguousarray(np.clip(np.rint(value / scale) + zero_point, 0, 255), dtype=np.uint8)


def dequantize(value, name):
    shape, scale, zero_point = ENCODINGS[name]
    if value.dtype != np.uint8 or value.shape != shape:
        raise ValueError(f"invalid native UINT8 {name} tensor")
    return (value.astype(np.float32) - zero_point) * np.float32(scale)


@dataclass(frozen=True)
class PreparedDiagnostic:
    native: np.ndarray
    width: int
    height: int
    scale_x: float
    scale_y: float
    clipped_fraction: float


def prepare_frame(bgr, *, allow_unverified_contract=False, preprocessing="mmdet_bgr_normalized"):
    if not allow_unverified_contract:
        raise ValueError("preprocessing is unverified for this context; explicit diagnostic opt-in required")
    preprocessing_contract(preprocessing)
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
        raise ValueError("expected nonempty uint8 HxWx3 BGR frame")
    import cv2
    height, width = bgr.shape[:2]
    ratio = min(640 / width, 640 / height)
    resized_w, resized_h = max(1, int(width * ratio + .5)), max(1, int(height * ratio + .5))
    image = np.full((640, 640, 3), 114, dtype=np.float32)
    image[:resized_h, :resized_w] = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    if preprocessing == "qai_rgb01":
        image = image[..., ::-1] / np.float32(255)
    else:
        image = (image - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
    floating = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    _, scale, zero = ENCODINGS["image"]
    scaled = floating / scale + zero
    clipped_fraction = float(np.mean((scaled < 0) | (scaled > 255)))
    return PreparedDiagnostic(quantize(floating), width, height, resized_w / width,
                              resized_h / height, clipped_fraction)


def decode_heads(outputs, prepared, *, confidence=.10, top_k=300, allow_unverified_contract=False):
    if not allow_unverified_contract:
        raise ValueError("six-head semantics are unverified; explicit diagnostic opt-in required")
    if not 0 < confidence <= 1 or type(top_k) is not int or not 1 <= top_k <= 300:
        raise ValueError("confidence must be >0..1 and top_k 1..300")
    if set(outputs) != set(ENCODINGS) - {"image"}:
        raise ValueError("expected six named RTMDet QNN output heads")
    boxes, scores = [], []
    for cls_name, reg_name, stride in HEADS:
        cls, reg = dequantize(outputs[cls_name], cls_name), dequantize(outputs[reg_name], reg_name)
        height, width = cls.shape[-2:]
        # Hypothesis: standard head logits and already stride-scaled LTRB distances.
        logits = cls.transpose(0, 2, 3, 1).reshape(1, -1, 80)
        distances = reg.transpose(0, 2, 3, 1).reshape(1, -1, 4)
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        priors = np.stack((xx, yy), -1).reshape(1, -1, 2).astype(np.float32) * stride
        boxes.append(np.concatenate((priors - distances[..., :2], priors + distances[..., 2:]), -1))
        scores.append(1 / (1 + np.exp(-np.clip(logits, -80, 80))))
    return rtmdet_decode({"boxes": np.concatenate(boxes, 1), "scores": np.concatenate(scores, 1)},
        width=prepared.width, height=prepared.height, scale_x=prepared.scale_x,
        scale_y=prepared.scale_y, confidence=confidence, top_k=top_k)


def read_native_outputs(output: Path, index: int):
    results = list(Path(output).rglob(f"Result_{index}"))
    if len(results) != 1 or not results[0].is_dir():
        raise ValueError(f"expected one Result_{index} directory, found {len(results)}")
    found = {}
    raw_paths = sorted(results[0].glob("*.raw"))
    for path in raw_paths:
        # QAIRT2.50 native dumps use <tensor>_native.raw. Match by tensor name,
        # never glob order or size: preserve plain dumps from older fixtures too.
        name = path.stem.removesuffix("_native")
        if name in found:
            raise ValueError(f"ambiguous native output names for {name}: {found[name].name}, {path.name}")
        found[name] = path
    if set(found) != set(ENCODINGS) - {"image"}:
        raise ValueError("native output names differ from pinned wrapper contract: "
                         + ", ".join(path.name for path in raw_paths))
    outputs = {}
    for name, path in found.items():
        shape, _, _ = ENCODINGS[name]
        if path.stat().st_size != int(np.prod(shape)):
            raise ValueError(f"{name} native UINT8 byte count mismatch; never interpret float output as UINT8")
        outputs[name] = np.fromfile(path, dtype=np.uint8).reshape(shape)
    return outputs


def runtime_paths(root: Path):
    root = Path(root).expanduser().resolve(strict=True)
    possibilities = (root, root / "qairt" / SDK_VERSION, root / SDK_VERSION)
    matches = [p for p in possibilities if (p / "bin/aarch64-ubuntu-gcc9.4/qnn-net-run").is_file()]
    if len(matches) != 1:
        raise ValueError("provide isolated runtime root containing exactly one Ubuntu ARM64 qnn-net-run")
    sdk = matches[0]
    runner = sdk / "bin/aarch64-ubuntu-gcc9.4/qnn-net-run"
    libs, dsp = sdk / "lib/aarch64-ubuntu-gcc9.4", sdk / "lib/hexagon-v68/unsigned"
    required = (runner, libs / "libQnnHtp.so", libs / "libQnnSystem.so",
                libs / "libQnnHtpV68Stub.so", dsp / "libQnnHtpV68Skel.so")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError("missing isolated runtime files: " + ", ".join(missing))
    return runner, libs, dsp


def run_native(folder, runtime_root, work, native, repeats=3, timeout_s=120):
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("QNN HTP diagnostic must run on Linux ARM64 Edge; no CPU fallback")
    if type(repeats) is not int or not 1 <= repeats <= 5 or not 0 < timeout_s <= 300:
        raise ValueError("diagnostic repeats1..5 and timeout0..300 required")
    if native.dtype != np.uint8 or native.shape != ENCODINGS["image"][0]:
        raise ValueError("native input must be UINT8 [1,3,640,640]")
    folder = verify_bundle(folder)
    runner, libs, dsp = runtime_paths(runtime_root)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(libs) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["ADSP_LIBRARY_PATH"] = str(dsp) + (";" + env["ADSP_LIBRARY_PATH"] if env.get("ADSP_LIBRARY_PATH") else "")
    work = Path(work).resolve(strict=True)
    # All writes exclusive in caller-owned fresh work directory; no cleanup/deletion.
    version = subprocess.run([str(runner), "--version"], env=env, cwd=work,
                             capture_output=True, text=True, timeout=15, check=True)
    version_text = version.stdout + version.stderr
    if SDK_VERSION not in version_text:
        raise RuntimeError("runner is not the expected compatible QAIRT2.50 runtime")
    with (work / "runner_version.txt").open("x", encoding="utf-8") as target:
        target.write(version_text)
    raw = work / "input_uint8.raw"
    with raw.open("xb") as target:
        target.write(native.tobytes())
    if any(ch.isspace() for ch in str(raw)):
        raise ValueError("QNN input-list path must not contain whitespace")
    with (work / "input_list.txt").open("x", encoding="utf-8", newline="\n") as target:
        target.write((str(raw) + "\n") * repeats)
    output = work / "output"
    if output.exists():
        raise FileExistsError("output already exists; choose a fresh diagnostic directory")
    command = [str(runner), "--backend", str(libs / "libQnnHtp.so"),
               "--retrieve_context", str(folder / "model.bin"),
               "--input_list", str(work / "input_list.txt"), "--output_dir", str(output),
               "--use_native_input_files", "--use_native_output_files", "--profiling_level", "basic"]
    with (work / "runner.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, cwd=work, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout_s)
    if result.returncode:
        raise RuntimeError(f"QNN HTP failed exit={result.returncode}; inspect {work / 'runner.log'}")
    # Success requires byte-valid outputs for every repeat, not just process exit0.
    outputs = [read_native_outputs(output, index) for index in range(repeats)]
    return outputs, {"command": command, "runner_version": version_text.strip(), "sdk_version": SDK_VERSION,
                     "runner_sha256": sha256(runner), "htp_sha256": sha256(libs / "libQnnHtp.so"),
                     "profile_files": [str(p) for p in sorted(output.rglob("*.log"))]}
