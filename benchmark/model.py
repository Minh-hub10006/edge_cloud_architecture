"""Unpaced model-adapter capacity, without pretending to measure pipeline FPS."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import monotonic_ns, time_ns
from typing import Any

from .config import RunConfig, finite_nonnegative
from .resources import ResourceSampler
from .summarize import summarize, write_report
from .trace import JsonlWriter, frame_event


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                            stderr=subprocess.DEVNULL, timeout=3, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        manifest = root / "RELEASE_MANIFEST.json"
        return {"commit": None, "dirty": None,
                "release_manifest_sha256": sha256(manifest) if manifest.is_file() else None}


def preload_video(path: Path, cache_frames: int, max_bytes: int) -> list[Any]:
    """Fixed first-N source frames, replayed cyclically; bounded and disclosed."""
    if cache_frames <= 0 or max_bytes <= 0:
        raise ValueError("positive cache limits required")
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    size = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        for _ in range(cache_frames):
            ok, frame = cap.read()
            if not ok:
                break
            size += frame.nbytes
            if size > max_bytes:
                raise ValueError("frame cache exceeds --cache-mib; reduce --cache-frames")
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError("video has no decoded frames")
    return frames


def load_adapter(kind: str, artifact: Path, *, imgsz: int = 320,
                 layout: str = "nchw", confidence: float = 0.25,
                 allow_diagnostic: bool = False, nms_iou: float = .5):
    if kind == "yolo_snpe":
        from inference.adapters import YoloSnpeAdapter
        detector = YoloSnpeAdapter(dlc_path=str(artifact), imgsz=imgsz,
                                  input_layout=layout, score_threshold=confidence)
        onnx = detector.detector.session.mode == "ort_cuda"
        runtime = {"backend": "onnxruntime" if onnx else "snpe", "device": detector.detector.session.runtime,
                   "session_mode": detector.detector.session.mode,
                   "actual_device_utilization_verified": False,
                   "gpu_providers": detector.detector.session.get_providers() if onnx else None,
                   "input_layout": layout, "imgsz": imgsz,
                   "confidence": confidence}
    elif kind == "dfine_onnx":
        from inference.adapters import DfineCloudAdapter
        detector = DfineCloudAdapter(str(artifact))
        runtime = {"backend": "onnxruntime", "device": detector.runtime,
                   "gpu_providers": detector.detector.sess.get_providers()}
        import os
        runtime.update(top_k=int(os.environ.get("DFINE_TOP_K", "20")),
                       confidence=float(os.environ.get("DFINE_CONF_THR", "0.10")),
                       box_format=os.environ.get("DFINE_BOX_FORMAT", "cxcywh"))
    elif kind in {"rtdetrv2_onnx", "lwdetr_onnx"}:
        from inference.cloud_detr import RtdetrV2SmallAdapter, LwDetrLargeAdapter
        cls = RtdetrV2SmallAdapter if kind == "rtdetrv2_onnx" else LwDetrLargeAdapter
        import os
        opt_in = "RTDETRV2_ALLOW_IMPORTED_CANDIDATE" if kind == "rtdetrv2_onnx" else "LWDETR_ALLOW_IMPORTED_CANDIDATE"
        detector = cls(str(artifact), confidence=confidence,
                       allow_imported_candidate=os.getenv(opt_in, "0") == "1")
        runtime = {"backend": "onnxruntime", "device": detector.runtime,
                   "gpu_providers": detector.session.get_providers(), **detector.settings}
    elif kind == "yolo11s_snpe":
        onnx = artifact.suffix.lower() == ".onnx"
        if imgsz != 320 or layout != ("nchw" if onnx else "nhwc"):
            raise ValueError("YOLO11s split-v8 runtime requires imgsz320 / NHWC DLC or NCHW ONNX")
        from inference.yolo11s_snpe import Yolo11sSnpeAdapter
        detector = Yolo11sSnpeAdapter(str(artifact), allow_diagnostic=allow_diagnostic,
                                     confidence=confidence, nms_iou=nms_iou)
        runtime = {"backend": "onnxruntime" if onnx else "snpe", "device": detector.session.runtime,
                   "actual_device_utilization_verified": False, **detector.settings}
    elif kind == "rtmdet_snpe":
        from inference.rtmdet_snpe import RtmdetSmallSnpeAdapter
        detector = RtmdetSmallSnpeAdapter(str(artifact), confidence=confidence)
        runtime = {"backend": "snpe", "device": detector.session.runtime,
                   "session_mode": detector.session.mode,
                   "actual_device_utilization_verified": False, **detector.settings}
    else:
        raise ValueError("unknown adapter")
    return detector, runtime


def infer_staged(detector: Any, frame: Any):
    """Full adapter call with its own stage split when the adapter exposes one."""
    if callable(getattr(detector, "infer_detailed", None)):
        return detector.infer_detailed(frame)
    if callable(getattr(detector, "prepare_frame", None)) and callable(getattr(detector, "infer_prepared", None)):
        return detector.infer_prepared(detector.prepare_frame(frame))
    boxes, reported = detector.infer(frame)
    return boxes, reported, {}


def measure_model(config: RunConfig, detector: Any, frames: list[Any], directory: Path,
                  *, identity: dict[str, Any], sample_s: float = 1.0,
                  clock=monotonic_ns, writer_capacity: int = 1024) -> dict[str, Any]:
    if config.profile != "model_capacity" or not frames:
        raise ValueError("model runner requires model capacity and cached frames")
    if writer_capacity <= 0:
        raise ValueError("writer capacity must be positive")
    if sample_s != 0 and (sample_s < 0.1 or not sample_s < float("inf")):
        raise ValueError("resource interval must be 0 (disabled) or finite >=0.1")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "config": config.as_dict(), "created_utc_ns": time_ns(),
                "code": source_revision(), "python": platform.python_version(),
                "host_platform": platform.platform(), "identity": identity,
                "workload": "first cached frames, cyclic replay; batch=1; unpaced",
                "cached_frames": len(frames), "tracking": False, "monitor": False,
                "completion": "adapter.infer returned normalized boxes",
                "stages": {"adapter_call_ms": "wall time around full adapter call (prepare+infer+decode)",
                           "adapter_reported_ms": "adapter-reported duration, not kernel-only",
                           "edge_*/cloud_*": "adapter's own stage split (preprocess/session/postprocess)"},
                "excluded": ["model load", "video read/decode", "RTSP", "transport", "tracking"],
                "resource_interval_s": sample_s, "writer_capacity": writer_capacity}
    with (directory / "manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2, allow_nan=False)
    writer = JsonlWriter(directory / "trace.jsonl", writer_capacity)
    sampler = None
    fatal = None
    measured_started = None
    measured_finished = None
    elapsed_s = 0.0
    interrupted: BaseException | None = None
    try:
        if sample_s:
            sampler = ResourceSampler(directory / "resources.jsonl", config.node_id, sample_s)
            sampler.start()
        for phase, count in (("warmup", config.warmup_frames), ("measure", config.measured_frames)):
            if phase == "measure":
                measured_started = clock()
            for index in range(count):
                begin = clock()
                status = "ok"
                error = None
                boxes = []
                reported = None
                detail = {}
                try:
                    boxes, reported, detail = infer_staged(detector, frames[index % len(frames)])
                    reported = finite_nonnegative(reported, "adapter reported ms")
                    if not isinstance(boxes, (list, tuple)):
                        raise ValueError("adapter returned invalid boxes container")
                except Exception as exc:
                    status = "error"
                    reported = None
                    error = f"{type(exc).__name__}: {exc}"
                end = clock()
                writer.write(frame_event(config, frame_id=index, phase=phase, status=status,
                                         started_ns=begin, finished_ns=end,
                                         durations_ms={"adapter_call_ms": (end - begin) / 1e6,
                                                       "adapter_reported_ms": reported,
                                                       **(detail if status == "ok" else {})},
                                         detections=len(boxes) if status == "ok" else 0, error=error))
                if phase == "warmup" and status != "ok":
                    raise RuntimeError(f"warmup failed: {error}")
            if phase == "measure":
                measured_finished = clock()
                elapsed_s = (measured_finished - measured_started) / 1e9
    except BaseException as exc:
        fatal = f"{type(exc).__name__}: {exc}"
        if measured_started is not None:
            measured_finished = clock()
            elapsed_s = (measured_finished - measured_started) / 1e9
        if not isinstance(exc, Exception):
            interrupted = exc
    finally:
        for resource in (sampler, writer):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    fatal = (fatal + "; " if fatal else "") + f"{type(exc).__name__}: {exc}"
    verified = bool(identity.get("artifact_sha256") and identity.get("video_sha256")
                    and identity.get("model_id") == getattr(detector, "model_id", None)
                    and identity.get("runtime"))
    try:
        summary = summarize(directory / "trace.jsonl", config, elapsed_s,
                            fatal_error=fatal, identity_verified=verified)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Preserve an explicit invalid report even when trace itself is damaged.
        summary = {"schema_version": 1, **config.as_dict(), "elapsed_s": elapsed_s,
                   "completed": None, "effective_fps": None, "stages_ms": {},
                   "gates": {"execution": False, "measurement": False,
                             "artifact_runtime_identity": verified, "quality": None},
                   "execution_status": "FAIL", "quality_status": "NOT_EVALUATED",
                   "fatal_error": f"{fatal or ''}; trace invalid: {type(exc).__name__}: {exc}"}
    summary["resource_samples"] = sampler.writer.written if sampler else 0
    summary["measurement_interval"] = {"started_mono_ns": measured_started,
                                       "finished_mono_ns": measured_finished,
                                       "clock_scope": config.node_id}
    summary["resources_scope"] = "filter resource mono_ns by measurement_interval; warmup excluded"
    summary["trace_written"] = writer.written
    write_report(directory, summary)
    if interrupted:
        raise interrupted
    return summary
