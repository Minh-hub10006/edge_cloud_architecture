"""ONNX Runtime session helpers — prefer QNN HTP on QCS6490, fallback CPU."""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None  # type: ignore


def pipeline_root() -> str:
    return os.environ.get(
        "PIPELINE_AI_DETECTION_ROOT",
        os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts")),
    )


def resolve_segment_a_onnx(explicit: Optional[str] = None) -> str:
    """FP32 backbone default for CPU; INT8 when QNN EP is available."""
    if explicit and os.path.isfile(explicit):
        return explicit
    for key in ("SEGMENT_A_ONNX",):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
    onnx_dir = os.path.join(art, "onnx")
    int8 = os.path.join(onnx_dir, "segment_a_int8.onnx")
    fp32 = os.path.join(onnx_dir, "segment_a.onnx")
    backend = os.environ.get("SEGMENT_A_BACKEND", "auto").strip().lower()
    if backend in ("qnn_so", "npu", "htp") and os.path.isfile(int8):
        return int8
    if HAS_ORT:
        available = set(ort.get_available_providers())
        if "QNNExecutionProvider" in available and os.path.isfile(int8):
            return int8
    if os.path.isfile(fp32):
        return fp32
    if os.path.isfile(int8):
        return int8
    root = pipeline_root()
    for name in ("segment_a.onnx", "segment_a_int8.onnx"):
        cand = os.path.join(root, "onnx_export", name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(root, "onnx_export", "segment_a.onnx")


def _qnn_backend_path() -> str:
    explicit = os.environ.get("QNN_BACKEND_PATH", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    for cand in (
        "/usr/lib/libQnnHtp.so",
        "/usr/lib/aarch64-linux-gnu/libQnnHtp.so",
    ):
        if os.path.isfile(cand):
            return cand
    backup = os.path.expanduser("~/backup_qnn_20260323_095238/qc/qairt")
    if os.path.isdir(backup):
        for ver in sorted(os.listdir(backup), reverse=True):
            lib = os.path.join(
                backup, ver, "lib", "aarch64-android", "libQnnHtp.so"
            )
            if os.path.isfile(lib):
                return lib
    sigm = os.path.expanduser("~/sigm/qairt")
    if os.path.isdir(sigm):
        for ver in sorted(os.listdir(sigm), reverse=True):
            for sub in ("aarch64-oe-linux-gcc11.2", "aarch64-ubuntu-gcc9.4"):
                lib = os.path.join(sigm, ver, "lib", sub, "libQnnHtp.so")
                if os.path.isfile(lib):
                    return lib
    return ""


def ensure_cuda_dlls() -> None:
    """Windows: put CUDA/cuDNN next to the ORT CUDA provider on the DLL search path.

    GPU edges may lack a full CUDA toolkit `bin/`; DLLs can ship in torch or in
    `nvidia-*` pip wheels (`nvidia/cublas/bin`, `nvidia/cudnn/bin`, ...).
    """
    if os.name != "nt" or getattr(ensure_cuda_dlls, "_done", False):
        return
    ensure_cuda_dlls._done = True  # type: ignore[attr-defined]
    dirs = [d for d in os.environ.get("ORT_CUDA_DLL_DIRS", "").split(os.pathsep) if d]
    try:
        import importlib.util
        import site

        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            dirs.append(os.path.join(os.path.dirname(spec.origin), "lib"))
        for site_dir in list(site.getsitepackages()) + ([site.getusersitepackages()] if site.getusersitepackages() else []):
            nvidia_root = os.path.join(site_dir, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            for pkg in sorted(os.listdir(nvidia_root)):
                for sub in (
                    "bin",
                    os.path.join("bin", "x64"),
                    os.path.join("bin", "x86_64"),
                    "lib",
                ):
                    cand = os.path.join(nvidia_root, pkg, sub)
                    if os.path.isdir(cand):
                        dirs.append(cand)
    except Exception:
        pass
    cuda_root = os.environ.get("CUDA_PATH", "").strip()
    if cuda_root:
        for sub in ("bin", os.path.join("bin", "x64")):
            dirs.append(os.path.join(cuda_root, sub))
    # De-dupe while preserving order.
    seen: set[str] = set()
    found: list[str] = []
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d))
        if key in seen or not os.path.isdir(d):
            continue
        seen.add(key)
        found.append(d)
    for d in found:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    # add_dll_directory alone is not enough: the ORT provider bridge resolves its
    # CUDA dependencies through the plain PATH search.
    if found:
        os.environ["PATH"] = os.pathsep.join(found + [os.environ.get("PATH", "")])


def ort_provider_chain(*, qnn_context_bin: Optional[str] = None) -> list[Any]:
    """Build provider list: QNN HTP (edge NPU) → CUDA (Windows GPU) → CPU."""
    if not HAS_ORT:
        return []
    ensure_cuda_dlls()
    force = os.environ.get("ORT_EXECUTION_PROVIDERS", "").strip()
    if force:
        parts = [p.strip() for p in force.split(",") if p.strip()]
        # Drop CPU-only force when CUDA is available and required — fleet used to
        # pin edge-247 to CPUExecutionProvider and tank FPS.
        avail = set(ort.get_available_providers())
        if (
            parts == ["CPUExecutionProvider"]
            and "CUDAExecutionProvider" in avail
            and os.environ.get("ORT_FORCE_CPU", "").strip().lower()
            not in ("1", "true", "yes", "on")
        ):
            print(
                "[ORT] ignoring ORT_EXECUTION_PROVIDERS=CPUExecutionProvider; "
                "CUDA is available (set ORT_FORCE_CPU=1 to keep CPU)"
            )
            parts = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return parts or ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    chain: list[Any] = []
    if "QNNExecutionProvider" in available:
        qnn_opts: dict[str, str] = {
            "htp_performance_mode": os.environ.get("QNN_HTP_PERF", "burst"),
        }
        backend = _qnn_backend_path()
        if backend:
            qnn_opts["backend_path"] = backend
        ctx_bin = (qnn_context_bin or os.environ.get("QNN_CONTEXT_BIN", "")).strip()
        if not ctx_bin:
            ctx_bin = os.environ.get("SEGMENT_A_QNN_BIN", "").strip()
        if not ctx_bin:
            art = os.environ.get("EDGE_ARTIFACTS", os.path.expanduser("~/edge_artifacts"))
            cand = os.path.join(art, "npu", "segment_a_qnn_int8.bin")
            if os.path.isfile(cand):
                ctx_bin = cand
        if ctx_bin and os.path.isfile(ctx_bin):
            qnn_opts["qnn_context_cache_path"] = ctx_bin
        chain.append(("QNNExecutionProvider", qnn_opts))
    if "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        chain.append("DmlExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


def _want_cuda(providers: list[Any]) -> bool:
    names = []
    for item in providers:
        names.append(item[0] if isinstance(item, tuple) else str(item))
    return any("CUDA" in n or "Tensorrt" in n for n in names)


def _require_cuda() -> bool:
    raw = os.environ.get("ORT_REQUIRE_CUDA", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # Windows ORT edges: CUDA is the expected path; silent CPU kills FPS.
    force = os.environ.get("ORT_EXECUTION_PROVIDERS", "").strip()
    if force and any("CUDA" in p or "Tensorrt" in p for p in force.split(",")):
        return True
    if os.name == "nt" and os.environ.get("YOLO_BACKEND", "ort").strip().lower() == "ort":
        return True
    return False


def create_ort_session(
    onnx_path: str,
    *,
    label: str = "ORT",
    qnn_context_bin: Optional[str] = None,
):
    """Create InferenceSession; prefer QNN/CUDA. Never silently land on CPU when CUDA is required."""
    if not HAS_ORT:
        raise ImportError("onnxruntime not installed")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    # Must run before InferenceSession so the CUDA EP can resolve cuDNN/cuBLAS.
    ensure_cuda_dlls()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ort_provider_chain(qnn_context_bin=qnn_context_bin)
    require_cuda = _require_cuda() and _want_cuda(providers)
    last_err: Optional[Exception] = None
    # Never fall back to CPU on INT8 ONNX (ConvInteger unsupported).
    is_int8 = onnx_path.endswith("_int8.onnx") or "int8" in os.path.basename(onnx_path)
    # Silent CPU fallback was the FPS killer on the RTX edge.
    allow_cpu_fallback = (
        not require_cuda
        and not is_int8
        and os.environ.get("ORT_ALLOW_CPU_FALLBACK", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    attempts: list[list[Any]] = [providers]
    if allow_cpu_fallback:
        attempts.append(["CPUExecutionProvider"])

    for attempt in attempts:
        if not attempt:
            continue
        try:
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=attempt)
            used = list(sess.get_providers())
            runtime = runtime_label(used)
            print(f"[{label}] {onnx_path}")
            print(f"[{label}] providers={used} runtime={runtime}")
            if require_cuda and not any("CUDA" in p or "Tensorrt" in p for p in used):
                avail = ort.get_available_providers()
                raise RuntimeError(
                    f"[{label}] ORT_REQUIRE_CUDA=1 but session landed on {used}. "
                    f"available={avail}. Install onnxruntime-gpu matching the CUDA toolkit, "
                    f"ensure torch cuDNN DLLs are on PATH (ensure_cuda_dlls), "
                    f"or set ORT_ALLOW_CPU_FALLBACK=1 only for debug."
                )
            if used and used[0] == "CPUExecutionProvider":
                print(
                    f"[{label}] WARNING running on CPU — FPS will be low. "
                    f"Set ORT_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider"
                )
            return sess, runtime, used
        except Exception as e:
            last_err = e
            if attempt == ["CPUExecutionProvider"]:
                break
    raise RuntimeError(f"ORT session failed for {onnx_path}: {last_err}")


def runtime_label(providers: list[str]) -> str:
    if not providers:
        return "unknown"
    p0 = providers[0]
    if "QNN" in p0:
        return "onnxruntime_qnn_htp"
    if p0 == "CPUExecutionProvider":
        return "onnxruntime_cpu"
    if "CUDA" in p0:
        return "onnxruntime_cuda"
    return p0.lower().replace("executionprovider", "")


def dfine_encode_mode() -> str:
    """mono | split | fdraas — default mono (high FPS LAN path for DRL).

    ``cut_stem`` is dead for primary use (2026-09-18): even with working
    qnn-daemon uint8 NCHW stem (~12 FPS), it loses to ``dfine_mono`` (~136 FPS)
    on the fair A/B. Opt-in only via ``DFINE_ALLOW_DEAD_CUT_STEM=1``.
    """
    mode = os.environ.get("DFINE_ENCODE", "mono").strip().lower()
    if mode in ("mono", "jpeg", "full", "monolithic"):
        return "mono"
    if mode in ("cut_stem", "stem", "cut-stem"):
        allow = os.environ.get("DFINE_ALLOW_DEAD_CUT_STEM", "0").strip().lower()
        if allow in ("1", "true", "yes", "on"):
            return "cut_stem"
        print(
            "[DFINE] cut_stem is deprecated/dead for primary paths "
            "(loses FPS vs mono; set DFINE_ALLOW_DEAD_CUT_STEM=1 to force). "
            "Falling back to mono."
        )
        return "mono"
    if mode in ("split", "tier1", "backbone", "segment_a"):
        return "split"
    if mode in ("fdraas", "0x02", "dist"):
        return "fdraas"
    return "mono"
