"""QNN Segment A inference — ORT QNN EP in-process or libsegment_a_int8.so."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from detectors.dfine.ort_providers import (
    HAS_ORT,
    create_ort_session,
    resolve_segment_a_onnx,
)
from detectors.dfine.preprocess import frame_to_images_tensor
from detectors.dfine.tier1_codec import (
    P3_SHAPE,
    P4_SHAPE,
    P5_SHAPE,
    Q3_SHAPE,
    Q4_SHAPE,
    Q5_SHAPE,
    validate_proj256_tensors,
    validate_tier1_tensors,
)
from detectors.npu.snpe_env import qairt_abi, qairt_root, setup_qairt_env

# libsegment_a_int8.so outputs NHWC layout_transform tensors (see probe_segment_a_qnn_layout.py)
_QNN_NHWC_SPECS: dict[str, tuple[str, tuple[int, ...], int]] = {
    "P3": ("stages_1", (1, 80, 80, 512), 384),
    "P4": ("stages_2", (1, 40, 40, 1024), 768),
    "P5": ("stages_3", (1, 20, 20, 2048), 1536),
}
_QNN_OUTPUT_PROFILES: dict[str, dict[str, tuple[str, tuple[int, ...], int, tuple[int, ...]]]] = {
    "backbone": {
        "P3": ("stages_1", (1, 80, 80, 512), 384, P3_SHAPE),
        "P4": ("stages_2", (1, 40, 40, 1024), 768, P4_SHAPE),
        "P5": ("stages_3", (1, 20, 20, 2048), 1536, P5_SHAPE),
    },
    "proj256": {
        "Q3": ("output_0", (1, 80, 80, 256), 256, Q3_SHAPE),
        "Q4": ("output_1", (1, 40, 40, 256), 256, Q4_SHAPE),
        "Q5": ("output_2", (1, 20, 20, 256), 256, Q5_SHAPE),
    },
}


def qnn_output_profile() -> str:
    profile = os.environ.get("QNN_OUTPUT_PROFILE", "backbone").strip().lower()
    if profile in ("p", "p345", "tier1"):
        return "backbone"
    if profile in ("proj", "projection", "proj256", "q345"):
        return "proj256"
    return profile


def _is_constant(arr: np.ndarray) -> bool:
    """True if the array is a flat/constant placeholder (std ~ 0)."""
    if arr.size == 0:
        return True
    try:
        return bool(np.ptp(arr) == 0)
    except Exception:
        return False


def resolve_segment_a_so(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for key in ("SEGMENT_A_QNN_SO", "SEGMENT_A_SO"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
    for sub in ("npu243", "npu"):
        cand = os.path.join(art, sub, "libsegment_a_int8.so")
        if os.path.isfile(cand):
            return cand
    return os.path.join(art, "npu", "libsegment_a_int8.so")


def segment_a_backend_mode() -> str:
    return os.environ.get("SEGMENT_A_BACKEND", "auto").strip().lower()


def qnn_subprocess_allowed() -> bool:
    """qnn-net-run per frame recompiles graph (~6s) — only for explicit microbench."""
    raw = os.environ.get("SEGMENT_A_QNN_SUBPROCESS", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def qnn_persistent_enabled() -> bool:
    raw = os.environ.get("QNN_PERSISTENT", os.environ.get("SNPE_PERSISTENT", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


class QnnOrtSegmentABackend:
    """Segment A via ONNX Runtime + QNN EP (session kept warm)."""

    def __init__(self, onnx_path: Optional[str] = None):
        if not HAS_ORT:
            raise ImportError("onnxruntime not installed")
        self.onnx_path = resolve_segment_a_onnx(onnx_path)
        self.session, self.runtime, self.providers = create_ort_session(
            self.onnx_path, label="QnnOrtSegmentA"
        )
        if "QNN" not in (self.providers[0] if self.providers else ""):
            raise RuntimeError(f"QNN EP not active: {self.providers}")

    def infer_tensors(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        t0 = time.perf_counter()
        images = frame_to_images_tensor(frame_bgr)
        in_name = self.session.get_inputs()[0].name
        outs = self.session.run(None, {in_name: images.astype(np.float32)})
        out_map = {o.name: v for o, v in zip(self.session.get_outputs(), outs)}
        if "P3" in out_map:
            tensors = {"P3": out_map["P3"], "P4": out_map["P4"], "P5": out_map["P5"]}
        else:
            tensors = {"P3": outs[0], "P4": outs[1], "P5": outs[2]}
        return time.perf_counter() - t0, tensors

    def close(self) -> None:
        self.session = None


class QnnNativeSegmentABackend:
    """Segment A via QAIRT model_level_api NativeExecutor (in-process, no per-frame spawn)."""

    def __init__(
        self,
        context_bin: Optional[str] = None,
        *,
        perf_profile: Optional[str] = None,
    ):
        self.context_bin = context_bin or os.environ.get("SEGMENT_A_QNN_CONTEXT", "").strip()
        if not self.context_bin:
            for cand in (
                os.path.expanduser("~/edge_artifacts/npu/segment_a_board.serialized.bin"),
                os.path.expanduser("~/edge_artifacts/npu/segment_a_qnn_int8.bin"),
            ):
                if os.path.isfile(cand):
                    self.context_bin = cand
                    break
        if not self.context_bin or not os.path.isfile(self.context_bin):
            raise FileNotFoundError(os.path.join("context", self.context_bin or ""))
        root = qairt_root()
        pyroot = os.environ.get(
            "QAIRT_PYTHON_ROOT", os.path.join(root, "lib", "python")
        )
        if pyroot not in sys.path:
            sys.path.insert(0, pyroot)
        from qti.aisw.core import model_level_api as mla
        from qti.aisw.tools.core.modules.api.backend.htp_backend import HtpBackend
        from qti.aisw.tools.core.modules.api.definitions import ModelConfig

        self._mla = mla
        self._perf_profile = perf_profile or os.environ.get("QNN_HTP_PERF", "burst")
        backend = HtpBackend(
            target=mla.OELinuxTarget(),
            config_dict={"perfProfile": self._perf_profile},
        )
        model = ModelConfig(path=self.context_bin)
        self._inferencer = mla.Inferencer(
            backend=backend, model=model, executor=mla.NativeExecutor()
        )
        # NOTE: do NOT use InferenceConfig(use_native_output_data=True) here —
        # it makes QAIRT return placeholder buffers (all 0 / const 17), which
        # broke Segment A (commit c42ef43). Keep the default float32 path so the
        # backend returns real dequantized tensors; the int8 wire codec does its
        # own symmetric per-tensor quantization.
        self.native_output_enabled = False
        self._run_config = None
        self._native_output_dir = "/tmp/qnn_seg_out"
        self._inferencer.setup(
            config=None, output_dir=self._native_output_dir
        )
        self._feature_guard_remaining = max(
            0, int(os.environ.get("DFINE_TIER1_GUARD_FRAMES", "3"))
        )

    def _run_raw(self, frame_bgr: np.ndarray) -> Tuple[float, dict]:
        t0 = time.perf_counter()
        images = frame_to_images_tensor(frame_bgr)
        out_dir = self._native_output_dir
        if os.path.isdir(out_dir):
            for p in Path(out_dir).rglob("*"):
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        raw = self._inferencer.run(
            np.ascontiguousarray(images, dtype=np.float32),
            "",
            config=self._run_config,
            output_dir=out_dir,
        )
        dt = time.perf_counter() - t0
        out_dict = raw[0] if raw else {}
        if not isinstance(out_dict, dict):
            raise RuntimeError(f"unexpected QNN native result type {type(out_dict)!r}")
        if os.environ.get("QNN_DEBUG_OUTPUTS", "0") == "1":
            files = [
                f"{str(p.relative_to(out_dir))}:{p.stat().st_size}"
                for p in sorted(Path(out_dir).rglob("*"), key=lambda p: p.name)
                if p.is_file()
            ]
            print(
                "[QNNNative] output_dir files: " + ", ".join(files) or "(none)",
                flush=True,
            )
        return dt, out_dict

    def _match_outputs(self, out_dict: dict) -> Dict[str, np.ndarray]:
        if os.environ.get("QNN_DEBUG_OUTPUTS", "0") == "1":
            print(
                "[QNNNative] raw output names/shapes: "
                + json.dumps(
                    {
                        name: (list(v.shape) if v is not None else None)
                        for name, v in out_dict.items()
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        outputs: Dict[str, np.ndarray] = {}
        expected_nchw = {"P3": P3_SHAPE, "P4": P4_SHAPE, "P5": P5_SHAPE}
        for key, (needle, nhwc_shape, _ch_out) in _QNN_NHWC_SPECS.items():
            match: Optional[np.ndarray] = None
            for name, value in out_dict.items():
                if value is None:
                    continue
                arr = np.asarray(value)
                # QAIRT versions differ: some return padded NHWC native tensors,
                # others return dequantized NCHW model outputs.
                if tuple(arr.shape) in (nhwc_shape, expected_nchw[key]):
                    match = arr
                    break
            if match is None:
                match = next(
                    (v for name, v in out_dict.items() if needle in name and v is not None),
                    None,
                )
            if match is not None:
                arr = np.asarray(match)
                # The in-memory buffers may be placeholders (all constant) when
                # use_native_output_data writes the real bytes to output files.
                if arr.size > 0 and _is_constant(arr):
                    from_file = self._read_native_file(nhwc_shape)
                    if from_file is not None:
                        arr = from_file
            else:
                from_file = self._read_native_file(nhwc_shape)
                if from_file is None:
                    raise RuntimeError(f"QNN native output for {key} not found")
                arr = from_file
            arr = np.asarray(arr)
            if tuple(arr.shape) not in (nhwc_shape, expected_nchw[key]):
                raise RuntimeError(
                    f"QNN native output {key} shape {tuple(arr.shape)} is neither "
                    f"{nhwc_shape} nor {expected_nchw[key]}"
                )
            outputs[key] = arr
        return outputs

    def _read_native_file(self, nhwc_shape: tuple) -> Optional[np.ndarray]:
        """Read one QNN native U8 tensor from the output dir, if written there."""
        nbytes = int(np.prod(nhwc_shape))
        for p in sorted(Path(self._native_output_dir).rglob("*"), key=lambda p: p.stat().st_size):
            if not p.is_file() or p.stat().st_size != nbytes:
                continue
            try:
                arr = np.fromfile(p, dtype=np.uint8).reshape(nhwc_shape)
            except (ValueError, OSError):
                continue
            if _is_constant(arr):
                continue
            if os.environ.get("QNN_DEBUG_OUTPUTS", "0") == "1":
                print(
                    f"[QNNNative] read {p.name} from file shape={nhwc_shape}",
                    flush=True,
                )
            return arr
        return None

    def infer_native_tensors(
        self, frame_bgr: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Return graph-native padded NHWC UINT8 tensors without dequantizing."""
        if not self.native_output_enabled:
            raise RuntimeError("QNN native output is disabled")
        dt, out_dict = self._run_raw(frame_bgr)
        outputs = self._match_outputs(out_dict)
        for key, arr in outputs.items():
            if arr.dtype != np.uint8:
                raise TypeError(f"QNN native output {key} dtype {arr.dtype} != uint8")
        return dt, outputs

    def infer_tensors(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        dt, out_dict = self._run_raw(frame_bgr)
        raw_outputs = self._match_outputs(out_dict)
        tensors: Dict[str, np.ndarray] = {}
        expected_nchw = {"P3": P3_SHAPE, "P4": P4_SHAPE, "P5": P5_SHAPE}
        for key, (_needle, nhwc_shape, ch_out) in _QNN_NHWC_SPECS.items():
            raw_arr = np.asarray(raw_outputs[key])
            if not np.issubdtype(raw_arr.dtype, np.floating):
                raise RuntimeError(
                    f"QNN Segment-A {key} returned {raw_arr.dtype} without verified "
                    "quantization metadata; refusing guessed dequantization"
                )
            arr = np.asarray(raw_arr, dtype=np.float32)
            if tuple(arr.shape) == expected_nchw[key]:
                nchw = arr
            elif tuple(arr.shape) == nhwc_shape:
                nchw = np.transpose(arr, (0, 3, 1, 2))[:, :ch_out, :, :]
            else:
                raise RuntimeError(f"unexpected {key} shape {tuple(arr.shape)}")
            tensors[key] = np.ascontiguousarray(nchw, dtype=np.float32)
            if tuple(tensors[key].shape) != expected_nchw[key]:
                raise RuntimeError(f"{key} shape {tensors[key].shape} != expected")
        if self._feature_guard_remaining:
            validate_tier1_tensors(tensors, source="QNN Segment-A")
            self._feature_guard_remaining -= 1
        return dt, tensors

    def close(self) -> None:
        try:
            self._inferencer.teardown()
        except Exception:
            pass
        self._inferencer = None


class QnnSoSegmentABackend:
    """Segment A via qnn-net-run + libsegment_a_int8.so (shared graph cache dir)."""

    def __init__(
        self,
        model_so: Optional[str] = None,
        *,
        perf_profile: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_so = resolve_segment_a_so(model_so)
        if not os.path.isfile(self.model_so):
            raise FileNotFoundError(self.model_so)
        root = qairt_root()
        abi = qairt_abi()
        self.qnn_bin = os.path.join(root, "bin", abi, "qnn-net-run")
        self.backend_so = os.environ.get(
            "QNN_BACKEND_PATH", os.path.join(root, "lib", abi, "libQnnHtp.so")
        )
        if not os.path.isfile(self.qnn_bin):
            raise FileNotFoundError(self.qnn_bin)
        self.perf_profile = perf_profile or os.environ.get("QNN_HTP_PERF", "burst")
        self.context_bin = os.environ.get("SEGMENT_A_QNN_CONTEXT", "").strip()
        if not self.context_bin:
            for cand in (
                os.path.expanduser("~/edge_artifacts/npu/segment_a_board.serialized.bin"),
                os.path.expanduser("~/edge_artifacts/npu/segment_a_qnn_int8.bin"),
            ):
                if os.path.isfile(cand):
                    self.context_bin = cand
                    break
        base = cache_dir or os.environ.get(
            "SEGMENT_A_QNN_CACHE", os.path.expanduser("~/edge_artifacts/segment_a243/pipe_cache")
        )
        self.cache_dir = Path(base)
        self.in_dir = self.cache_dir / "in"
        self.out_dir = self.cache_dir / "out"
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._input_raw = self.in_dir / "input.raw"
        self._input_list = self.in_dir / "input_list.txt"
        self._input_list.write_text(f"{self._input_raw}\n", encoding="utf-8")

    def infer_tensors(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        t0 = time.perf_counter()
        images = frame_to_images_tensor(frame_bgr)
        self._input_raw.write_bytes(np.ascontiguousarray(images, dtype=np.float32).tobytes())
        if self.out_dir.exists():
            for p in self.out_dir.glob("**/*"):
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        cmd = [
            self.qnn_bin,
        ]
        if self.context_bin:
            cmd += ["--retrieve_context", self.context_bin]
        else:
            cmd += ["--model", self.model_so]
        cmd += [
            "--backend",
            self.backend_so,
            "--input_list",
            str(self._input_list),
            "--output_dir",
            str(self.out_dir),
            "--num_inferences",
            "1",
            "--keep_num_outputs",
            "1",
            "--perf_profile",
            self.perf_profile,
            "--log_level",
            "error",
        ]
        subprocess.run(
            cmd,
            check=True,
            env=setup_qairt_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tensors = _parse_qnn_outputs(self.out_dir)
        return time.perf_counter() - t0, tensors

    def close(self) -> None:
        pass


class QnnDaemonSegmentABackend:
    """Segment A via the persistent native qnn-daemon (Unix socket).

    The qnn-daemon keeps the HTP context binary + graph loaded, so inference is
    ~18ms/frame with real (dequantized) features — unlike the broken in-process
    NativeExecutor (which returns all-zero placeholders on QAIRT 2.43).
    """

    def __init__(
        self,
        *,
        socket_path: Optional[str] = None,
        context_bin: Optional[str] = None,
        backend_so: Optional[str] = None,
        system_so: Optional[str] = None,
    ):
        from detectors.npu.qnn_daemon_client import QnnDaemonClient

        self.socket_path = socket_path or os.environ.get(
            "QNN_DAEMON_SOCKET", "/tmp/qnn_daemon.sock"
        )
        daemon_bin = os.environ.get("QNN_DAEMON_BIN", "")
        self.context_bin = context_bin or os.environ.get("SEGMENT_A_QNN_CONTEXT", "").strip()
        if not self.context_bin:
            for cand in (
                os.path.expanduser("~/edge_artifacts/npu/segment_a_board.serialized.bin"),
                os.path.expanduser("~/edge_artifacts/npu/segment_a_qnn_int8.bin"),
            ):
                if os.path.isfile(cand):
                    self.context_bin = cand
                    break
        if not self.context_bin or not os.path.isfile(self.context_bin):
            raise FileNotFoundError(os.path.join("context", self.context_bin or ""))

        root = qairt_root()
        abi_dir = os.path.join(root, "lib", qairt_abi())
        self.backend_so = backend_so or os.environ.get(
            "QNN_BACKEND_PATH", os.path.join(abi_dir, "libQnnHtp.so")
        )
        self.system_so = system_so or os.environ.get(
            "QNN_SYSTEM_PATH", os.path.join(abi_dir, "libQnnSystem.so")
        )
        self.shm_dir = os.environ.get(
            "QNN_DAEMON_SHM_DIR", "/dev/shm/qnn_segment_a"
        )
        os.makedirs(self.shm_dir, exist_ok=True)
        self.input_bits = int(os.environ.get("QNN_INPUT_BITS", "8"))
        if self.input_bits not in (8, 16):
            raise ValueError(
                f"unsupported QNN_INPUT_BITS={self.input_bits}; expected 8 or 16"
            )
        self.input_bytes = int(np.prod((1, 3, 640, 640))) * (self.input_bits // 8)
        self.output_layout = os.environ.get("QNN_OUTPUT_LAYOUT", "NHWC").strip().upper()
        if self.output_layout not in ("NCHW", "NHWC"):
            raise ValueError(
                f"unsupported QNN_OUTPUT_LAYOUT={self.output_layout}; expected NCHW or NHWC"
            )
        self.output_profile = qnn_output_profile()
        if self.output_profile not in _QNN_OUTPUT_PROFILES:
            raise ValueError(
                f"unsupported QNN_OUTPUT_PROFILE={self.output_profile}; "
                f"expected one of {sorted(_QNN_OUTPUT_PROFILES)}"
            )
        self.output_specs = _QNN_OUTPUT_PROFILES[self.output_profile]
        self._feature_guard_remaining = max(
            0, int(os.environ.get("DFINE_TIER1_GUARD_FRAMES", "3"))
        )

        self._client: Optional[QnnDaemonClient] = None
        self._daemon_proc = None

        # wait for daemon socket to appear (daemon must be started externally)
        import time
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if os.path.exists(self.socket_path):
                # socket exists, create our single persistent client
                from detectors.npu.qnn_daemon_client import QnnDaemonClient
                self._client = QnnDaemonClient(
                    self.socket_path,
                    shm_dir=self.shm_dir,
                    input_bytes=self.input_bytes,
                )
                if self._client.ping():
                    break
                self._client.close()
                self._client = None
            if daemon_bin and os.path.isfile(daemon_bin) and self._daemon_proc is None:
                self._spawn_daemon(daemon_bin)
            time.sleep(0.1)
        if self._client is None:
            raise RuntimeError(
                f"QNN daemon not reachable at {self.socket_path} "
                f"(start it or set QNN_DAEMON_BIN/QNN_DAEMON_SOCKET)"
            )
        self.native_output_enabled = True

    def _ping(self) -> bool:
        try:
            client = QnnDaemonClient(self.socket_path)
            ok = client.ping()
            if ok:
                self._client = client
            else:
                client.close()
            return ok
        except Exception:
            return False

    def _spawn_daemon(self, daemon_bin: str) -> None:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            os.path.dirname(self.backend_so) + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        try:
            self._daemon_proc = subprocess.Popen(
                [
                    daemon_bin,
                    "--context-bin", self.context_bin,
                    "--socket", self.socket_path,
                    "--backend", self.backend_so,
                    "--system", self.system_so,
                    "--shm-dir", self.shm_dir,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"could not launch qnn-daemon: {exc}") from exc

        # wait for socket file to appear + daemon init (~200ms)
        import time
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if os.path.exists(self.socket_path):
                time.sleep(0.2)  # extra margin for "ready" log
                return
            time.sleep(0.05)
        raise RuntimeError("qnn-daemon did not create socket within 5s")

    def _ensure_client(self):
        from detectors.npu.qnn_daemon_client import QnnDaemonClient

        if self._client is not None:
            return self._client
        self._client = QnnDaemonClient(
            self.socket_path,
            shm_dir=self.shm_dir,
            input_bytes=self.input_bytes,
        )
        if not self._client.ping():
            raise RuntimeError(f"QNN daemon not reachable at {self.socket_path}")
        return self._client

    def infer_tensors(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Return dequantized float32 output-profile tensors in NCHW layout."""
        from detectors.npu.qnn_daemon_client import QnnDaemonClient
        from detectors.dfine.preprocess import frame_to_images_tensor

        # The canonical INT8 context exposes uint8 input; the opt-in W8A16
        # candidate exposes uint16. Both use the full unsigned image range.
        images = frame_to_images_tensor(frame_bgr)
        # QAIRT serializes this HTP graph's native input buffer physically as
        # NHWC even though the source ONNX input is NCHW.  Sending NCHW bytes
        # still executes successfully, but scrambles spatial/channel data and
        # produces unusable detections.  The board layout diagnostic verifies
        # NHWC input together with logical NHWC P3/P4/P5 outputs.
        images_nhwc = np.transpose(images, (0, 2, 3, 1))
        if self.input_bits == 8:
            native_input = np.ascontiguousarray(
                np.clip(np.rint(images_nhwc * 255.0), 0, 255), dtype=np.uint8
            )
        else:
            native_input = np.ascontiguousarray(
                np.clip(np.rint(images_nhwc * 65535.0), 0, 65535), dtype=np.uint16
            )
        client = self._ensure_client()
        t0 = time.perf_counter()
        dt_graph, raw = client.infer_bytes(native_input.tobytes())
        dt = time.perf_counter() - t0

        outputs: Dict[str, np.ndarray] = {}
        for key, (needle, nhwc_shape, ch_out, expected_nchw) in self.output_specs.items():
            want_padded = int(np.prod(nhwc_shape))
            logical_nhwc = (1, nhwc_shape[1], nhwc_shape[2], ch_out)
            want_logical = int(np.prod(logical_nhwc))
            arr = None
            is_padded = False
            for value in raw.values():
                a = np.asarray(value, dtype=np.float32).ravel()
                if a.size == want_padded:
                    if self.output_layout == "NHWC":
                        arr = np.transpose(
                            a.reshape(nhwc_shape), (0, 3, 1, 2)
                        )
                    else:
                        arr = a.reshape(
                            (1, nhwc_shape[3], nhwc_shape[1], nhwc_shape[2])
                        )
                    is_padded = True
                    break
                if a.size == want_logical:
                    if self.output_layout == "NHWC":
                        arr = np.transpose(
                            a.reshape(logical_nhwc), (0, 3, 1, 2)
                        )
                    else:
                        arr = a.reshape(
                            (1, ch_out, nhwc_shape[1], nhwc_shape[2])
                        )
                    break
            if arr is None:
                raise RuntimeError(f"QNN daemon output for {key} not found")
            nchw = arr
            if is_padded:
                nchw = nchw[:, :ch_out, :, :]
            outputs[key] = np.ascontiguousarray(nchw, dtype=np.float32)
            if tuple(outputs[key].shape) != expected_nchw:
                raise RuntimeError(
                    f"QNN daemon output {key} shape {outputs[key].shape} != {expected_nchw}"
                )
        if self._feature_guard_remaining:
            if self.output_profile == "proj256":
                validate_proj256_tensors(outputs, source="QNN daemon Segment-A proj256")
            else:
                validate_tier1_tensors(outputs, source="QNN daemon Segment-A")
            self._feature_guard_remaining -= 1
        return max(dt, dt_graph), outputs

    def infer_packed_int8(
        self, frame_bgr: np.ndarray, affine: Dict[str, tuple[float, float]]
    ) -> Tuple[float, bytes]:
        """Use daemon command 6: W8A16 graph -> affine -> Tier-1 INT8, all native.

        The caller must explicitly enable this path after validating the exact
        W8A16 NCHW context.  There is intentionally no guessed layout fallback.
        """
        if self.output_profile != "backbone":
            raise RuntimeError("native Tier1 packing is only valid for QNN_OUTPUT_PROFILE=backbone")
        if self.input_bits != 16 or self.output_layout != "NCHW":
            raise RuntimeError("native Tier1 packing requires QNN_INPUT_BITS=16 and QNN_OUTPUT_LAYOUT=NCHW")
        from detectors.dfine.preprocess import frame_to_images_tensor

        images = frame_to_images_tensor(frame_bgr)
        images_nhwc = np.transpose(images, (0, 2, 3, 1))
        native_input = np.ascontiguousarray(
            np.clip(np.rint(images_nhwc * 65535.0), 0, 65535), dtype=np.uint16
        )
        coeffs: list[float] = []
        for key in ("P3", "P4", "P5"):
            if key not in affine:
                raise RuntimeError(f"native Tier1 packing missing affine for {key}")
            scale, bias = affine[key]
            coeffs.extend((float(scale), float(bias)))
        client = self._ensure_client()
        t0 = time.perf_counter()
        dt_graph, packed = client.infer_tier1_packed_shm(
            native_input.tobytes(), tuple(coeffs)
        )
        from detectors.dfine.tier1_codec import TIER1_BYTES_INT8, TIER1_INT8_HEADER_BYTES
        expected = TIER1_INT8_HEADER_BYTES + TIER1_BYTES_INT8
        if len(packed) != expected:
            raise RuntimeError(f"native Tier1 packed payload {len(packed)} != {expected}")
        return max(time.perf_counter() - t0, dt_graph), packed

    def infer_packed_proj256_int8(
        self, frame_bgr: np.ndarray, affine: Dict[str, tuple[float, float]]
    ) -> Tuple[float, bytes]:
        """Use daemon command 7: W8A16 projection graph -> Q3/Q4/Q5 INT8.

        Unlike ``infer_tensors``, this keeps the 2.15-M element output out of
        Python entirely.  It is opt-in and validates both the selected graph
        profile and exact wire length before returning bytes to gRPC.
        """
        if self.output_profile != "proj256":
            raise RuntimeError("native proj256 packing requires QNN_OUTPUT_PROFILE=proj256")
        if self.input_bits != 16 or self.output_layout != "NCHW":
            raise RuntimeError("native proj256 packing requires QNN_INPUT_BITS=16 and QNN_OUTPUT_LAYOUT=NCHW")
        from detectors.dfine.preprocess import frame_to_images_tensor
        from detectors.dfine.tier1_codec import (
            PROJ256_BYTES_INT8,
            PROJ256_INT8_HEADER_BYTES,
        )

        images = frame_to_images_tensor(frame_bgr)
        images_nhwc = np.transpose(images, (0, 2, 3, 1))
        native_input = np.ascontiguousarray(
            np.clip(np.rint(images_nhwc * 65535.0), 0, 65535), dtype=np.uint16
        )
        coeffs: list[float] = []
        for key in ("Q3", "Q4", "Q5"):
            if key not in affine:
                raise RuntimeError(f"native proj256 packing missing affine for {key}")
            scale, bias = affine[key]
            coeffs.extend((float(scale), float(bias)))
        client = self._ensure_client()
        t0 = time.perf_counter()
        dt_graph, packed = client.infer_proj256_packed_shm(
            native_input.tobytes(), tuple(coeffs)
        )
        expected = PROJ256_INT8_HEADER_BYTES + PROJ256_BYTES_INT8
        if len(packed) != expected:
            raise RuntimeError(f"native proj256 packed payload {len(packed)} != {expected}")
        return max(time.perf_counter() - t0, dt_graph), packed

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
                self._client = None
            except Exception:
                self._client = None
        if self._daemon_proc is not None:
            try:
                self._daemon_proc.terminate()
            except Exception:
                pass


def _parse_qnn_outputs(out_dir: Path) -> Dict[str, np.ndarray]:
    raw_files = list(out_dir.glob("**/*.raw"))
    if not raw_files:
        raise RuntimeError(f"no QNN raw outputs under {out_dir}")
    outputs: Dict[str, np.ndarray] = {}
    for key, (needle, nhwc_shape, ch_out) in _QNN_NHWC_SPECS.items():
        matches = [f for f in raw_files if needle in f.name]
        if not matches:
            nbytes = int(np.prod(nhwc_shape)) * 4
            matches = [f for f in raw_files if f.stat().st_size == nbytes]
        if not matches:
            raise RuntimeError(f"QNN output for {key} not found under {out_dir}")
        arr = np.fromfile(matches[0], dtype=np.float32).reshape(nhwc_shape)
        nchw = np.transpose(arr, (0, 3, 1, 2))[:, :ch_out, :, :]
        outputs[key] = np.ascontiguousarray(nchw, dtype=np.float32)
        if tuple(outputs[key].shape) != {"P3": P3_SHAPE, "P4": P4_SHAPE, "P5": P5_SHAPE}[key]:
            raise RuntimeError(f"{key} shape {outputs[key].shape} != expected")
    return outputs


def try_create_qnn_backend(
    *,
    prefer: str = "auto",
    onnx_path: Optional[str] = None,
    model_so: Optional[str] = None,
) -> Optional[Tuple[object, str]]:
    """Return (backend, mode_label) or None."""
    mode = prefer if prefer != "auto" else segment_a_backend_mode()
    order: list[str]
    daemon_bin = os.environ.get("QNN_DAEMON_BIN", "").strip()
    daemon_ok = bool(daemon_bin and os.path.isfile(daemon_bin))
    if mode == "ort_qnn":
        order = ["ort_qnn"]
    elif mode == "qnn_daemon":
        order = ["qnn_daemon"]
    elif mode in ("qnn_so", "so", "npu", "htp"):
        # Prefer qnn-daemon: QAIRT 2.43 NativeExecutor returns all-zero P3 placeholders.
        if daemon_ok:
            order = ["qnn_daemon"]
        else:
            order = ["qnn_so"] if qnn_subprocess_allowed() else ["qnn_so", "ort_qnn"]
    elif mode == "cpu":
        return None
    else:
        # auto: daemon first when available, else native / ort_qnn / subprocess .so
        order = ["qnn_daemon"] if daemon_ok else ["qnn_native"]
        if qnn_subprocess_allowed():
            order += ["qnn_so", "ort_qnn"]
        else:
            order.append("ort_qnn")
        if not daemon_ok and mode != "auto" and "qnn_native" not in order:
            order.insert(0, "qnn_native")

    for kind in order:
        try:
            if kind == "ort_qnn":
                backend = QnnOrtSegmentABackend(onnx_path=onnx_path)
                return backend, "qnn_ort"
            if kind == "qnn_so":
                if qnn_subprocess_allowed():
                    backend = QnnSoSegmentABackend(model_so=model_so)
                    return backend, "qnn_so"
                backend = QnnNativeSegmentABackend()
                return backend, "qnn_native"
            if kind == "qnn_native":
                backend = QnnNativeSegmentABackend()
                return backend, "qnn_native"
            if kind == "qnn_daemon":
                backend = QnnDaemonSegmentABackend()
                return backend, "qnn_daemon"
        except Exception as exc:
            print(f"[QnnSegmentA] {kind} unavailable ({exc})")
    return None
