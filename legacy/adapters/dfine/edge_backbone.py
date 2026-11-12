"""
D-FINE backbone on edge = Segment A (HGNetV2 → P3, P4, P5).

NPU path priority (SEGMENT_A_BACKEND=auto):
  1. QNN persistent worker (ORT QNN EP or libsegment_a_int8.so)
  2. ORT session (QNN EP / CPU FP32 fallback)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from detectors.dfine.ort_providers import (
    HAS_ORT,
    create_ort_session,
    resolve_segment_a_onnx,
)
from detectors.dfine.preprocess import frame_to_images_tensor
from detectors.dfine.tier1_codec import (
    flatten_tier1,
    mock_tier1,
    pack_proj256_int8,
    pack_tier1_int8,
    proj256_affine_coeffs,
    tier1_uplink_int8,
    validate_proj256_tensors,
)
from detectors.npu.qnn_segment_a_backend import segment_a_backend_mode


def pipeline_root() -> str:
    return os.environ.get(
        "PIPELINE_AI_DETECTION_ROOT",
        os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts")),
    )


def default_segment_a_onnx() -> str:
    return resolve_segment_a_onnx()


class EdgeBackbone:
    """Segment A only — backbone for split / Cut-1 path (NPU when available)."""

    def __init__(self, onnx_path: Optional[str] = None):
        self.onnx_path = resolve_segment_a_onnx(onnx_path)
        self.session = None
        self.runtime = "mock"
        self.providers: List[str] = []
        self.infer_mode = "mock"
        self._qnn_worker = None
        self._qnn_direct = None
        self.last_uplink_layout = "tier1_int8"
        self.split_profile = os.environ.get("DFINE_SPLIT_PROFILE", "backbone").strip().lower()
        if self.split_profile in ("proj", "projection", "proj256", "q345"):
            self.split_profile = "proj256"
        elif self.split_profile in ("p", "p345", "tier1"):
            self.split_profile = "backbone"
        self._tier1_affine_path = os.environ.get("DFINE_TIER1_AFFINE_PATH", "").strip()
        self._tier1_affine = self._load_tier1_affine(self._tier1_affine_path)
        self.require_qnn = os.environ.get("DFINE_REQUIRE_QNN", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        backend = segment_a_backend_mode()
        if self.require_qnn and backend == "cpu":
            raise RuntimeError(
                "DFINE_REQUIRE_QNN=1 is incompatible with SEGMENT_A_BACKEND=cpu"
            )
        if backend != "cpu":
            self._try_qnn_paths(onnx_path)

        if self.session is None and self._qnn_worker is None and self._qnn_direct is None:
            if self.require_qnn:
                raise RuntimeError(
                    "D-FINE split requires QNN, but no QNN Segment-A backend started; "
                    "CPU fallback is disabled"
                )
            self._try_ort_cpu(onnx_path)

        if self.session is None and self._qnn_worker is None and self._qnn_direct is None:
            print(
                f"[EdgeBackbone] segment_a not found ({self.onnx_path}). "
                "Set SEGMENT_A_ONNX or run export_onnx.py in pipeline repo."
            )

    def _try_qnn_paths(self, onnx_path: Optional[str]) -> None:
        from detectors.npu.qnn_segment_a_backend import (
            qnn_persistent_enabled,
            try_create_qnn_backend,
            segment_a_backend_mode,
        )
        from detectors.npu.qnn_segment_a_worker import try_get_qnn_worker

        backend_mode = segment_a_backend_mode()
        # qnn_daemon IS the persistent process — no worker needed
        if backend_mode == "qnn_daemon":
            created = try_create_qnn_backend(onnx_path=onnx_path or self.onnx_path)
            if created is not None:
                self._qnn_direct, mode = created
                self.runtime = f"qnn_htp_{mode}"
                self.infer_mode = mode
                print(f"[EdgeBackbone] QNN direct mode={mode}")
                return
            return

        if qnn_persistent_enabled():
            worker = try_get_qnn_worker(onnx_path=onnx_path or self.onnx_path)
            if worker is not None:
                self._qnn_worker = worker
                self.runtime = f"qnn_htp_{worker.mode}"
                self.infer_mode = worker.mode
                print(f"[EdgeBackbone] QNN persistent worker mode={worker.mode}")
                return

        created = try_create_qnn_backend(onnx_path=onnx_path or self.onnx_path)
        if created is not None:
            self._qnn_direct, mode = created
            self.runtime = f"qnn_htp_{mode}"
            self.infer_mode = mode
            print(f"[EdgeBackbone] QNN direct mode={mode}")

    def _try_ort_cpu(self, onnx_path: Optional[str]) -> None:
        path = resolve_segment_a_onnx(onnx_path)
        if "int8" in os.path.basename(path):
            art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
            fp32 = os.path.join(art, "onnx", "segment_a.onnx")
            if os.path.isfile(fp32):
                path = fp32
        if not HAS_ORT or not os.path.isfile(path):
            return
        try:
            self.session, self.runtime, self.providers = create_ort_session(
                path, label="EdgeBackbone"
            )
            self.infer_mode = self.runtime
            self.onnx_path = path
        except Exception as exc:
            print(f"[EdgeBackbone] ORT fallback failed ({exc})")

    def forward(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Returns (seconds, {P3, P4, P5})."""
        if self._qnn_worker is not None:
            return self._qnn_worker.infer_tensors(frame_bgr)
        if self._qnn_direct is not None:
            return self._qnn_direct.infer_tensors(frame_bgr)

        t0 = time.time()
        if self.session is not None:
            images = frame_to_images_tensor(frame_bgr)
            in_name = self.session.get_inputs()[0].name
            outs = self.session.run(None, {in_name: images.astype(np.float32)})
            out_map = {o.name: v for o, v in zip(self.session.get_outputs(), outs)}
            if "P3" in out_map:
                P3, P4, P5 = out_map["P3"], out_map["P4"], out_map["P5"]
            else:
                P3, P4, P5 = outs[0], outs[1], outs[2]
        else:
            time.sleep(0.008)
            _, tensors = mock_tier1()
            P3, P4, P5 = tensors["P3"], tensors["P4"], tensors["P5"]
        return time.time() - t0, {"P3": P3, "P4": P4, "P5": P5}

    def forward_flat(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray | bytes]:
        if tier1_uplink_int8():
            total_s, payload, _timings = self.forward_packed_int8(frame_bgr)
            return total_s, payload
        t, tdict = self.forward(frame_bgr)
        return t, flatten_tier1(tdict["P3"], tdict["P4"], tdict["P5"])

    def forward_packed_int8(
        self, frame_bgr: np.ndarray
    ) -> Tuple[float, bytes, Dict[str, float | bool | str | None]]:
        """Return the tier1 INT8 payload, packing inside the QNN worker when possible."""
        affine_enabled = self._tier1_affine is not None
        native_pack_enabled = os.environ.get(
            "DFINE_QNN_DAEMON_NATIVE_PACK", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self.split_profile == "proj256":
            if native_pack_enabled:
                if self._qnn_direct is None or not callable(
                    getattr(self._qnn_direct, "infer_packed_proj256_int8", None)
                ):
                    raise RuntimeError(
                        "DFINE_QNN_DAEMON_NATIVE_PACK requires qnn-daemon command 7 for proj256"
                    )
                total_s, payload = self._qnn_direct.infer_packed_proj256_int8(
                    frame_bgr, proj256_affine_coeffs()
                )
                self.last_uplink_layout = "proj256_int8"
                return total_s, payload, {
                    "qnn_queue_wait_s": 0.0,
                    "qnn_inference_s": float(total_s),
                    "worker_pack_s": 0.0,
                    "worker_ipc_s": 0.0,
                    "edge_inference_s": float(total_s),
                    "packed_in_worker": True,
                    "native_qnn_daemon_pack": True,
                    "proj256_affine": True,
                    "uplink_layout": "proj256_int8",
                }
            inference_started = time.perf_counter()
            backend_s, tensors = self.forward(frame_bgr)
            inference_call_s = time.perf_counter() - inference_started
            validate_proj256_tensors(tensors, source="EdgeBackbone proj256")
            pack_started = time.perf_counter()
            payload = pack_proj256_int8(tensors["Q3"], tensors["Q4"], tensors["Q5"])
            pack_s = time.perf_counter() - pack_started
            self.last_uplink_layout = "proj256_int8"
            return inference_call_s + pack_s, payload, {
                "qnn_queue_wait_s": 0.0,
                "qnn_inference_s": float(backend_s),
                "worker_pack_s": 0.0,
                "worker_ipc_s": max(0.0, inference_call_s - float(backend_s)),
                "edge_inference_s": inference_call_s,
                "parent_pack_s": pack_s,
                "packed_in_worker": False,
                "tier1_affine": False,
                "proj256_affine": True,
                "uplink_layout": "proj256_int8",
            }
        if native_pack_enabled:
            if not affine_enabled or self._qnn_direct is None or not callable(
                getattr(self._qnn_direct, "infer_packed_int8", None)
            ):
                raise RuntimeError(
                    "DFINE_QNN_DAEMON_NATIVE_PACK requires QNN daemon and a validated affine artifact"
                )
            total_s, payload = self._qnn_direct.infer_packed_int8(
                frame_bgr, self._tier1_affine
            )
            self.last_uplink_layout = "tier1_int8"
            return total_s, payload, {
                "qnn_queue_wait_s": 0.0,
                "qnn_inference_s": float(total_s),
                "worker_pack_s": 0.0,
                "worker_ipc_s": 0.0,
                "edge_inference_s": float(total_s),
                "packed_in_worker": True,
                "native_qnn_daemon_pack": True,
                "tier1_affine": True,
                "uplink_layout": "tier1_int8",
            }
        worker_pack_enabled = os.environ.get(
            "DFINE_WORKER_PACK_INT8", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        # Correction must run before INT8 packing; never silently bypass it in worker packing.
        if affine_enabled:
            worker_pack_enabled = False
        if worker_pack_enabled and self._qnn_worker is not None and callable(
            getattr(self._qnn_worker, "infer_packed_int8", None)
        ):
            result = self._qnn_worker.infer_packed_int8(frame_bgr)
            self.last_uplink_layout = result.layout
            timings: Dict[str, float | bool | str | None] = {
                "qnn_queue_wait_s": result.queue_wait_s,
                "qnn_inference_s": result.inference_s,
                "worker_pack_s": result.pack_s,
                "worker_ipc_s": result.ipc_s,
                "ipc_input_send_s": result.ipc_input_send_s,
                "ipc_input_handoff_s": result.ipc_input_handoff_s,
                "worker_dispatch_s": result.worker_dispatch_s,
                "ipc_output_handoff_s": result.ipc_output_handoff_s,
                "ipc_unattributed_s": result.ipc_unattributed_s,
                "worker_shm_write_s": result.worker_shm_write_s,
                "parent_shm_read_s": result.parent_shm_read_s,
                "output_transport": result.output_transport,
                "edge_inference_s": (
                    result.queue_wait_s + result.inference_s + result.ipc_s
                ),
                "packed_in_worker": True,
                "uplink_layout": result.layout,
                "native_qnn_output": result.layout == "tier1_uint8_native",
            }
            return result.total_s, result.payload, timings

        inference_started = time.perf_counter()
        backend_s, tensors = self.forward(frame_bgr)
        inference_call_s = time.perf_counter() - inference_started
        pack_started = time.perf_counter()
        if affine_enabled:
            tensors = self._apply_tier1_affine(tensors, self._tier1_affine)
        payload = pack_tier1_int8(tensors["P3"], tensors["P4"], tensors["P5"])
        self.last_uplink_layout = "tier1_int8"
        pack_s = time.perf_counter() - pack_started
        timings = {
            "qnn_queue_wait_s": 0.0,
            "qnn_inference_s": float(backend_s),
            "worker_pack_s": 0.0,
            "worker_ipc_s": max(0.0, inference_call_s - float(backend_s)),
            "edge_inference_s": inference_call_s,
            "parent_pack_s": pack_s,
            "packed_in_worker": False,
            "tier1_affine": affine_enabled,
        }
        return inference_call_s + pack_s, payload, timings

    @staticmethod
    def _load_tier1_affine(artifact_path: str) -> Optional[Dict[str, tuple[float, float]]]:
        """Load and validate the optional scalar correction once, outside the frame loop."""
        if not artifact_path:
            return None
        try:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                coeffs = {}
                for key in ("P3", "P4", "P5"):
                    scale = np.asarray(artifact[f"{key}_scale"], dtype=np.float32)
                    bias = np.asarray(artifact[f"{key}_bias"], dtype=np.float32)
                    if scale.shape != (1,) or bias.shape != (1,) or not np.isfinite(scale).all() or not np.isfinite(bias).all():
                        raise RuntimeError(f"invalid global affine coefficients for {key}")
                    coeffs[key] = (float(scale[0]), float(bias[0]))
        except Exception as exc:
            raise RuntimeError(f"cannot load DFINE_TIER1_AFFINE_PATH={artifact_path}: {exc}") from exc
        print(f"[EdgeBackbone] Tier1 global affine: {artifact_path}")
        return coeffs

    @staticmethod
    def _apply_tier1_affine(
        tensors: Dict[str, np.ndarray], coeffs: Dict[str, tuple[float, float]]
    ) -> Dict[str, np.ndarray]:
        """Apply prevalidated scalar correction immediately before wire quantization."""
        corrected: Dict[str, np.ndarray] = {}
        for key in ("P3", "P4", "P5"):
            scale, bias = coeffs[key]
            value = np.asarray(tensors[key], dtype=np.float32) * scale + bias
            if not np.isfinite(value).all():
                raise RuntimeError(f"non-finite corrected Tier1 tensor {key}")
            corrected[key] = value.astype(np.float32, copy=False)
        return corrected
