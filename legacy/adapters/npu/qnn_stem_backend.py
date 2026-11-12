"""QNN Stem cut inference — persistent NativeExecutor or ORT fallback."""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

from detectors.dfine.ort_providers import HAS_ORT, create_ort_session
from detectors.dfine.preprocess import frame_to_images_tensor
from detectors.dfine.stem_codec import STEM_ELEM_COUNT, STEM_SHAPE
from detectors.npu.snpe_env import qairt_abi, qairt_root, setup_qairt_env


def stem_qairt_root() -> str:
    """AI Hub stem .bin requires QAIRT 2.45+ (board pin 2.43 is for legacy segment_a .so)."""
    return os.environ.get(
        "STEM_QAIRT_ROOT",
        "/opt/qcom/qairt/2.45.40.260406",
    )


def setup_stem_qairt_env() -> dict[str, str]:
    root = stem_qairt_root()
    env = os.environ.copy()
    # Match working bench (bench_edge_stage2_25_retrieve.sh / npu_stem)
    lib = os.environ.get(
        "STEM_QNN_LIB",
        f"{root}/lib/aarch64-ubuntu-gcc9.4",
    )
    skel = os.environ.get(
        "STEM_QNN_SKEL",
        f"{root}/lib/hexagon-v68/unsigned",
    )
    py = f"{root}/lib/python"
    env["QAIRT_ROOT"] = root
    env["LD_LIBRARY_PATH"] = f"{lib}:/usr/lib/aarch64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
    env["ADSP_LIBRARY_PATH"] = f"{skel};/usr/lib/rfsa/adsp;/usr/lib"
    if os.path.isdir(py):
        env["PYTHONPATH"] = f"{py}:{env.get('PYTHONPATH', '')}"
    return env


def resolve_stem_qnn_bin(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for key in ("STEM_QNN_BIN", "EDGE_STEM_QNN_BIN"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    for cand in (
        os.path.expanduser("~/cuts_e2e/edge_stem_int8.bin"),
        os.path.expanduser("~/edge_artifacts/npu/edge_stem_int8.bin"),
    ):
        if os.path.isfile(cand):
            return cand
    return os.path.expanduser("~/cuts_e2e/edge_stem_int8.bin")


def resolve_edge_stem_onnx(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for key in ("EDGE_STEM_ONNX", "STEM_EDGE_ONNX"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    for cand in (
        os.path.expanduser("~/cuts_e2e/edge_stem.onnx"),
        os.path.expanduser("~/dfine_cloud_cuts/edge_stem.onnx"),
    ):
        if os.path.isfile(cand):
            return cand
    return os.path.expanduser("~/cuts_e2e/edge_stem.onnx")


def _select_native_stem_output(outputs: dict) -> tuple[str, np.ndarray]:
    """Select exactly one verified stem tensor from a QNN output map.

    Earlier code selected the first sufficiently large output and truncated it.
    That can silently route a non-stem tensor to Cloud and produce zero boxes.
    """
    requested = os.environ.get("STEM_QNN_OUTPUT_NAME", "").strip()
    available: list[tuple[str, np.ndarray]] = []
    for key, value in outputs.items():
        if value is not None:
            available.append((str(key), np.asarray(value, dtype=np.float32)))
    if requested:
        selected = next(((key, array) for key, array in available if key == requested), None)
        if selected is None:
            names = ", ".join(f"{key}:{tuple(array.shape)}" for key, array in available)
            raise RuntimeError(f"STEM_QNN_OUTPUT_NAME={requested!r} not found; outputs={names}")
    else:
        exact = [(key, array) for key, array in available if array.size == STEM_ELEM_COUNT]
        if len(exact) != 1:
            names = ", ".join(f"{key}:{tuple(array.shape)}" for key, array in available)
            raise RuntimeError("cannot identify unique stem output; set STEM_QNN_OUTPUT_NAME. " f"Expected exactly one tensor with {STEM_ELEM_COUNT} values; outputs={names}")
        selected = exact[0]
    key, array = selected
    if array.size != STEM_ELEM_COUNT:
        raise RuntimeError(f"QNN output {key!r} has {array.size} values, expected {STEM_ELEM_COUNT}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"QNN output {key!r} contains NaN/Inf")
    if float(np.ptp(array)) < 1e-7:
        value = float(array.reshape(-1)[0])
        raise RuntimeError(
            f"QNN output {key!r} shape={tuple(array.shape)} is constant "
            f"at {value:.8g}; refusing invalid stem payload"
        )
    if tuple(array.shape) == (1, 160, 160, 32):
        stem = array.transpose(0, 3, 1, 2)
    else:
        stem = array.reshape(STEM_SHAPE)
    return key, np.ascontiguousarray(stem, dtype=np.float32)

def qnn_stem_persistent_enabled() -> bool:
    raw = os.environ.get("QNN_PERSISTENT", os.environ.get("SNPE_PERSISTENT", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")

def _frame_to_stem_uint8_nchw(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame → uint8 NCHW [1,3,640,640] for HTP stem context.

    Working benches and qnn-daemon use images_u8.raw (NCHW uint8), not float32.
    NHWC uint8 also executes but scrambles features (std mismatch vs ORT).
    """
    images = frame_to_images_tensor(frame_bgr)
    return np.ascontiguousarray(
        np.clip(np.rint(images * 255.0), 0, 255), dtype=np.uint8
    )


def _stem_daemon_paths() -> tuple[str, str, str]:
    """Return (backend_so, system_so, lib_dir) pinned to STEM QAIRT 2.45."""
    root = stem_qairt_root()
    lib = os.environ.get(
        "STEM_QNN_LIB", os.path.join(root, "lib", "aarch64-ubuntu-gcc9.4")
    )
    backend = os.environ.get(
        "STEM_QNN_BACKEND_PATH", os.path.join(lib, "libQnnHtp.so")
    )
    system = os.environ.get(
        "STEM_QNN_SYSTEM_PATH", os.path.join(lib, "libQnnSystem.so")
    )
    return backend, system, lib


class QnnNativeStemBackend:
    """Stem cut via QAIRT NativeExecutor (in-process, no per-frame spawn)."""

    def __init__(self, context_bin: Optional[str] = None, *, perf_profile: Optional[str] = None):
        self.context_bin = context_bin or resolve_stem_qnn_bin()
        if not os.path.isfile(self.context_bin):
            raise FileNotFoundError(self.context_bin)
        root = stem_qairt_root()
        pyroot = os.environ.get("STEM_QAIRT_PYTHON_ROOT", os.path.join(root, "lib", "python"))
        if pyroot not in sys.path:
            sys.path.insert(0, pyroot)
        # Apply QAIRT 2.45 libs before loading NativeExecutor
        for key, val in setup_stem_qairt_env().items():
            os.environ[key] = val
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
        self._inferencer.setup(config=None, output_dir="/tmp/qnn_stem_out")

    def infer_stem(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run stem via NativeExecutor.

        Input contract matches working ``images_u8.raw`` benches: uint8 NCHW.
        Note: on this board NativeExecutor still returns constant placeholders
        (same class of bug as Seg-A); prefer ``QnnDaemonStemBackend``.
        """
        t0 = time.perf_counter()
        input_layout = os.environ.get("STEM_QNN_INPUT_LAYOUT", "NCHW").strip().upper()
        input_dtype = os.environ.get("STEM_QNN_INPUT_DTYPE", "uint8").strip().lower()
        images = frame_to_images_tensor(frame_bgr)
        if input_layout == "NHWC":
            images = images.transpose(0, 2, 3, 1)
        elif input_layout != "NCHW":
            raise RuntimeError(
                f"unsupported STEM_QNN_INPUT_LAYOUT={input_layout!r}; expected NCHW or NHWC"
            )
        if input_dtype in ("uint8", "u8"):
            qnn_input = np.ascontiguousarray(
                np.clip(np.rint(images * 255.0), 0, 255), dtype=np.uint8
            )
        elif input_dtype in ("float32", "float", "fp32"):
            qnn_input = np.ascontiguousarray(images, dtype=np.float32)
        else:
            raise RuntimeError(
                f"unsupported STEM_QNN_INPUT_DTYPE={input_dtype!r}; expected uint8 or float32"
            )
        raw = self._inferencer.run(
            qnn_input,
            "",
            output_dir="/tmp/qnn_stem_out",
        )
        dt = time.perf_counter() - t0
        out_dict = raw[0] if raw else {}
        if not out_dict:
            raise RuntimeError("QNN stem native returned no outputs")
        output_name, stem = _select_native_stem_output(out_dict)
        if not getattr(self, "_output_contract_logged", False):
            self._output_contract_logged = True
            print(
                f"[QnnStem] input_layout={input_layout} dtype={input_dtype} "
                f"output={output_name} shape={tuple(stem.shape)} "
                f"min={float(stem.min()):.5g} max={float(stem.max()):.5g} "
                f"mean={float(stem.mean()):.5g} std={float(stem.std()):.5g}"
            )
        return dt, stem
    def close(self) -> None:
        try:
            self._inferencer.teardown()
        except Exception:
            pass
        self._inferencer = None

class QnnDaemonStemBackend:
    """Stem cut via persistent native qnn-daemon (Unix socket + shm).

    NativeExecutor returns constant stem placeholders on this board (same class
    of bug as Seg-A). qnn-daemon keeps the HTP context loaded and accepts the
    real uint8 NCHW input used by legacy ``images_u8.raw`` benches (~8–9 ms).
    """

    def __init__(
        self,
        context_bin: Optional[str] = None,
        *,
        socket_path: Optional[str] = None,
    ):
        import subprocess

        from detectors.npu.qnn_daemon_client import QnnDaemonClient

        self.context_bin = context_bin or resolve_stem_qnn_bin()
        if not os.path.isfile(self.context_bin):
            raise FileNotFoundError(self.context_bin)

        self.socket_path = socket_path or os.environ.get(
            "STEM_QNN_DAEMON_SOCKET", "/tmp/qnn_daemon_stem.sock"
        )
        self.shm_dir = os.environ.get("STEM_QNN_DAEMON_SHM_DIR", "/dev/shm/qnn_stem")
        os.makedirs(self.shm_dir, exist_ok=True)
        self.input_bytes = int(np.prod((1, 3, 640, 640)))  # uint8 NCHW
        self.backend_so, self.system_so, self.lib_dir = _stem_daemon_paths()
        daemon_bin = os.environ.get(
            "QNN_DAEMON_BIN", os.path.expanduser("~/edge_artifacts/bin/qnn-daemon")
        )

        self._client: Optional[object] = None
        self._daemon_proc = None
        self._output_contract_logged = False

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if os.path.exists(self.socket_path):
                client = QnnDaemonClient(
                    self.socket_path,
                    shm_dir=self.shm_dir,
                    input_bytes=self.input_bytes,
                )
                if client.ping():
                    self._client = client
                    break
                client.close()
            if (
                daemon_bin
                and os.path.isfile(daemon_bin)
                and self._daemon_proc is None
            ):
                self._spawn_daemon(daemon_bin)
            time.sleep(0.1)
        if self._client is None:
            raise RuntimeError(
                f"QNN stem daemon not reachable at {self.socket_path} "
                f"(start it or set QNN_DAEMON_BIN / STEM_QNN_DAEMON_SOCKET)"
            )

    def _spawn_daemon(self, daemon_bin: str) -> None:
        import subprocess

        env = setup_stem_qairt_env()
        env["LD_LIBRARY_PATH"] = (
            self.lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        try:
            self._daemon_proc = subprocess.Popen(
                [
                    daemon_bin,
                    "--context-bin",
                    self.context_bin,
                    "--socket",
                    self.socket_path,
                    "--backend",
                    self.backend_so,
                    "--system",
                    self.system_so,
                    "--shm-dir",
                    self.shm_dir,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"could not launch stem qnn-daemon: {exc}") from exc

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if os.path.exists(self.socket_path):
                time.sleep(0.2)
                return
            time.sleep(0.05)
        raise RuntimeError("stem qnn-daemon did not create socket within 5s")

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
            raise RuntimeError(f"QNN stem daemon not reachable at {self.socket_path}")
        return self._client

    def infer_stem(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        native_input = _frame_to_stem_uint8_nchw(frame_bgr)
        client = self._ensure_client()
        t0 = time.perf_counter()
        dt_graph, raw = client.infer_bytes(native_input.tobytes())
        dt = time.perf_counter() - t0
        output_name, stem = _select_native_stem_output(raw)
        if not self._output_contract_logged:
            self._output_contract_logged = True
            print(
                f"[QnnStemDaemon] input=uint8_nchw output={output_name} "
                f"shape={tuple(stem.shape)} "
                f"min={float(stem.min()):.5g} max={float(stem.max()):.5g} "
                f"mean={float(stem.mean()):.5g} std={float(stem.std()):.5g} "
                f"graph_ms={dt_graph * 1000:.2f}"
            )
        return max(dt, float(dt_graph)), stem

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._daemon_proc is not None:
            try:
                self._daemon_proc.terminate()
            except Exception:
                pass
            self._daemon_proc = None


class QnnRetrieveStemBackend:
    """Stem via qnn-net-run --retrieve_context (QAIRT 2.45 AI Hub bin)."""

    _STEM_RAW_SHAPE = (1, 32, 160, 160)
    _STEM_RAW_N = int(np.prod(_STEM_RAW_SHAPE))

    def __init__(self, context_bin: Optional[str] = None, *, cache_dir: Optional[str] = None):
        self.context_bin = context_bin or resolve_stem_qnn_bin()
        if not os.path.isfile(self.context_bin):
            raise FileNotFoundError(self.context_bin)
        root = stem_qairt_root()
        abi = qairt_abi()
        self.qnn_bin = os.path.join(root, "bin", abi, "qnn-net-run")
        self.backend_so = os.environ.get(
            "STEM_QNN_BACKEND_PATH",
            os.path.join(stem_qairt_root(), "lib", "aarch64-ubuntu-gcc9.4", "libQnnHtp.so"),
        )
        if not os.path.isfile(self.qnn_bin):
            raise FileNotFoundError(self.qnn_bin)
        base = cache_dir or os.environ.get(
            "STEM_QNN_CACHE", os.path.expanduser("~/cuts_e2e/npu_stem/pipe_cache")
        )
        from pathlib import Path

        self.cache_dir = Path(base)
        self.in_dir = self.cache_dir / "in"
        self.out_dir = self.cache_dir / "out"
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._input_raw = self.in_dir / "input.raw"
        self._input_list = self.in_dir / "input_list.txt"
        self._input_list.write_text(f"{self._input_raw}\n", encoding="utf-8")
        self._output_raw: Optional[Path] = None

    def infer_stem(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        import subprocess
        from pathlib import Path

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
            "--retrieve_context",
            self.context_bin,
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
            os.environ.get("QNN_HTP_PERF", "burst"),
            "--log_level",
            "error",
        ]
        subprocess.run(
            cmd,
            check=True,
            env=setup_stem_qairt_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raw_files = list(self.out_dir.glob("**/*.raw"))
        if not raw_files:
            raise RuntimeError(f"no QNN stem outputs under {self.out_dir}")
        if self._output_raw is None:
            nbytes = self._STEM_RAW_N * 4
            matches = [f for f in raw_files if f.stat().st_size == nbytes]
            self._output_raw = matches[0] if matches else raw_files[0]
        stem = np.fromfile(self._output_raw, dtype=np.float32).reshape(self._STEM_RAW_SHAPE)
        return time.perf_counter() - t0, np.ascontiguousarray(stem, dtype=np.float32)

    def close(self) -> None:
        pass


class OrtStemBackend:
    """Stem cut via ONNX Runtime (CPU/CUDA dev fallback)."""

    def __init__(self, onnx_path: Optional[str] = None):
        if not HAS_ORT:
            raise ImportError("onnxruntime not installed")
        self.onnx_path = resolve_edge_stem_onnx(onnx_path)
        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(self.onnx_path)
        self.session, self.runtime, self.providers = create_ort_session(
            self.onnx_path, label="OrtStem"
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer_stem(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        t0 = time.perf_counter()
        images = frame_to_images_tensor(frame_bgr)
        outs = self.session.run(None, {self.input_name: images.astype(np.float32)})
        stem = np.ascontiguousarray(outs[0], dtype=np.float32).reshape(STEM_SHAPE)
        return time.perf_counter() - t0, stem

    def close(self) -> None:
        self.session = None


def try_create_stem_backend(
    *,
    prefer: str = "auto",
    context_bin: Optional[str] = None,
    onnx_path: Optional[str] = None,
) -> Optional[Tuple[object, str]]:
    """Return (backend, mode_label) or None.

    Prefer qnn-daemon (uint8 NCHW): NativeExecutor returns constant placeholders
    on this board; retrieve is correct but ~600 ms/frame.
    """
    mode = prefer if prefer != "auto" else os.environ.get("STEM_BACKEND", "auto").strip().lower()
    order: list[str]
    if mode in ("daemon", "qnn_daemon"):
        order = ["qnn_daemon"]
    elif mode in ("npu", "qnn", "htp", "native"):
        # "npu" historically meant NativeExecutor; prefer daemon first (Seg-A pattern).
        order = ["qnn_daemon", "qnn_native"]
    elif mode in ("retrieve", "qnn_retrieve"):
        order = ["qnn_retrieve"]
    elif mode in ("ort", "cpu", "cuda"):
        order = ["ort"]
    else:
        order = ["qnn_daemon", "qnn_native", "qnn_retrieve", "ort"]

    for kind in order:
        try:
            if kind == "qnn_daemon":
                backend = QnnDaemonStemBackend(context_bin=context_bin)
                return backend, "qnn_daemon_stem"
            if kind == "qnn_native":
                backend = QnnNativeStemBackend(context_bin=context_bin)
                return backend, "qnn_native_stem"
            if kind == "qnn_retrieve":
                backend = QnnRetrieveStemBackend(context_bin=context_bin)
                return backend, "qnn_retrieve_stem"
            if kind == "ort":
                backend = OrtStemBackend(onnx_path=onnx_path)
                return backend, f"ort_stem_{backend.runtime}"
        except Exception as exc:
            print(f"[QnnStem] {kind} unavailable ({exc})")
    return None
