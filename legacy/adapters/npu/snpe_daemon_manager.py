"""Start and reuse native snpe-daemon processes (one per DLC)."""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
from typing import Dict, Optional

from legacy.adapters.npu.snpe_daemon_client import SnpeDaemonClient
from legacy.adapters.npu.snpe_env import setup_qairt_env, setup_yolo11s_candidate_env, snpe_persistent_enabled

_LOCK = threading.Lock()
_DAEMONS: Dict[str, "SnpeDaemonHandle"] = {}


class SnpeDaemonHandle:
    def __init__(
        self,
        dlc_path: str,
        *,
        socket_path: str,
        runtime: str,
        perf_profile: str,
        bin_path: str,
        output_tensors: Optional[tuple[str, ...]] = None,
        input_bytes: Optional[int] = None,
        namespace: str = "",
        graph_name: Optional[str] = None,
    ):
        self.dlc_path = dlc_path
        self.socket_path = socket_path
        self.runtime = runtime
        self.perf_profile = perf_profile
        self.bin_path = bin_path
        self.output_tensors = output_tensors
        self.graph_name = graph_name
        self.shm_dir = shm_dir_for_dlc(dlc_path, namespace=namespace)
        self.input_bytes = input_bytes if input_bytes is not None else default_input_bytes(dlc_path)
        self.proc: Optional[subprocess.Popen] = None
        self.client = SnpeDaemonClient(
            socket_path,
            shm_dir=self.shm_dir,
            input_bytes=self.input_bytes,
        )
        self._start()

    def _start(self) -> None:
        env =setup_yolo11s_candidate_env() if self.graph_name is not None else setup_qairt_env()
        if self.graph_name is not None:
            contract = subprocess.run([self.bin_path, "--contract-version"],
                env=env, capture_output=True, text=True, timeout=10, check=False)
            if contract.returncode != 0 or contract.stdout.strip() != "SNPE_DAEMON_GRAPH_DSP_V3":
                raise RuntimeError("graph-scoped DSP requires cache-aware candidate daemon v3; legacy binary rejected")
        if os.path.exists(self.socket_path):
            # Daemons intentionally outlive short benchmark processes.  Reuse
            # a healthy cross-process daemon instead of unlinking its live
            # socket and starting a duplicate process for the same DLC.
            # A daemon busy serving other streams can miss one ping: retry before declaring it dead (Edge246, 4
            # streams, Oct 28: one missed ping -> socket unlinked, duplicate daemon, every stream lost the DSP).
            if any(self.client.ping() or time.sleep(1.0) for _ in range(5)):
                print(
                    f"[SnpeDaemon] reused {self.dlc_path} -> {self.socket_path}"
                )
                return
            os.unlink(self.socket_path)
        os.makedirs(self.shm_dir, exist_ok=True)
        cmd = [
            self.bin_path,
            "--dlc",
            self.dlc_path,
            "--socket",
            self.socket_path,
            "--shm-dir",
            self.shm_dir,
            "--runtime",
            self.runtime,
            "--perf",
            self.perf_profile,
            "--no-init-cache" if self.graph_name is not None else "--init-cache",
        ]
        outs = (",".join(self.output_tensors) if self.output_tensors is not None
                else (os.environ.get("SNPE_OUTPUT_TENSORS") or "").strip())
        if not outs and "split" in os.path.basename(self.dlc_path).lower():
            outs = "boxes,scores"
        if outs:
            cmd.extend(["--outputs", outs])
        if self.graph_name is not None:
            cmd.extend(["--graph", self.graph_name])
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if self.client.ping():
                print(f"[SnpeDaemon] ready {self.dlc_path} -> {self.socket_path}")
                return
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() if self.proc.stderr else b"").decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(f"snpe-daemon exited early: {err}")
            time.sleep(0.2)
        raise RuntimeError("snpe-daemon ping timeout")

    def close(self) -> None:
        if self.proc is None:
            return  # reused another process's daemon: shutting it down or unlinking its socket kills its streams
        try:
            self.client.shutdown()
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        # The shm folder is per process (namespace has the pid): left behind, /dev/shm (RAM) grew every run (246, 8/11: 30 MB).
        import shutil
        shutil.rmtree(self.shm_dir, ignore_errors=True)


def _daemon_bin(graph_name: Optional[str] = None) -> str:
    if graph_name is not None:  # graph-scoped = cache-off v3 daemon (YOLO11s; YOLOv8n where YOLOV8N_SNPE_GRAPH is set)
        candidate = os.environ.get("YOLO11S_SNPE_DAEMON_BIN", "").strip()
        if candidate:
            if not os.path.isfile(candidate):
                raise FileNotFoundError("YOLO11S_SNPE_DAEMON_BIN not found: " + candidate)
            return candidate
    explicit = os.environ.get("SNPE_DAEMON_BIN", "").strip()
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError("explicit SNPE_DAEMON_BIN not found: " + explicit)
        return explicit
    cand = os.path.expanduser("~/edge_artifacts/bin/snpe-daemon")
    if os.path.isfile(cand):
        return cand
    return cand


def shm_dir_for_dlc(dlc_path: str, *, namespace: str = "") -> str:
    base = os.environ.get("SNPE_DAEMON_SOCK_DIR", "/dev/shm/snpe_daemon")
    os.makedirs(base, exist_ok=True)
    digest = hashlib.sha1((os.path.abspath(dlc_path) + namespace).encode("utf-8")).hexdigest()[:12]
    return os.path.join(base, f"shm_{digest}")


def default_input_bytes(dlc_path: str) -> int:
    """YOLO 640 NCHW float32 — matches typical yolov8n_640 DLC input."""
    name = os.path.basename(dlc_path).lower()
    if "320" in name:
        return 1 * 3 * 320 * 320 * 4
    return 1 * 3 * 640 * 640 * 4


def socket_path_for_dlc(dlc_path: str, *, namespace: str = "") -> str:
    base = os.environ.get("SNPE_DAEMON_SOCK_DIR", "/dev/shm/snpe_daemon")
    os.makedirs(base, exist_ok=True)
    digest = hashlib.sha1((os.path.abspath(dlc_path) + namespace).encode("utf-8")).hexdigest()[:12]
    return os.path.join(base, f"dlc_{digest}.sock")


def get_daemon_handle(
    dlc_path: str,
    *,
    runtime: str = "dsp",
    perf_profile: str = "burst",
    output_tensors: Optional[tuple[str, ...]] = None,
    input_bytes: Optional[int] = None,
    graph_name: Optional[str] = None,
) -> SnpeDaemonHandle:
    key = f"{os.path.abspath(dlc_path)}|{runtime}|{perf_profile}"
    namespace = ""
    if output_tensors is not None or input_bytes is not None or graph_name is not None:
        if (not output_tensors or len(set(output_tensors)) != len(output_tensors)
                or any(not name or "," in name for name in output_tensors)
                or type(input_bytes) is not int or input_bytes <= 0 or input_bytes % 4):
            raise ValueError("explicit daemon contract requires unique outputs and float input bytes")
        namespace = repr((runtime, perf_profile, output_tensors, input_bytes))
        if graph_name is not None:
            if runtime != "dsp" or not graph_name or not all(c.isalnum() or c in "_.-" for c in graph_name):
                raise ValueError("graph-scoped daemon requires DSP and a valid graph name")
            namespace += repr(("graph-v3-cache-off", graph_name, os.path.abspath(_daemon_bin(graph_name))))
        key += namespace
    with _LOCK:
        handle = _DAEMONS.get(key)
        if handle is not None:
            # An attached cross-process daemon has no Popen handle here, so
            # liveness must be determined by the protocol ping.
            if handle.client.ping():
                return handle
            try:
                handle.close()
            except Exception:
                pass
        bin_path = _daemon_bin(graph_name)
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(
                f"snpe-daemon not found ({bin_path}). Run scripts/build_snpe_daemon_edge.sh"
            )
        # The daemon serves one connection at a time and each stream keeps its connection open, so a daemon cannot
        # be shared across stream processes: others' pings block, they unlink the socket and start duplicates
        # (Edge246/252, 3-4 streams, Oct 28). One daemon per process; shared only by threads in it.
        namespace += f"|pid{os.getpid()}"
        sock = socket_path_for_dlc(dlc_path, namespace=namespace)
        handle = SnpeDaemonHandle(
            dlc_path,
            socket_path=sock,
            runtime=runtime,
            perf_profile=perf_profile,
            bin_path=bin_path,
            output_tensors=output_tensors,
            input_bytes=input_bytes,
            namespace=namespace,
            graph_name=graph_name,
        )
        _DAEMONS[key] = handle
        return handle


def try_get_daemon_client(
    dlc_path: str,
    *,
    runtime: str = "dsp",
    perf_profile: str = "burst",
    output_tensors: Optional[tuple[str, ...]] = None,
    input_bytes: Optional[int] = None,
    graph_name: Optional[str] = None,
) -> Optional[SnpeDaemonClient]:
    if not snpe_persistent_enabled():
        return None
    if os.environ.get("SNPE_DAEMON", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        options = {} if output_tensors is None and input_bytes is None else {
            "output_tensors": output_tensors, "input_bytes": input_bytes}
        if graph_name is not None:
            options["graph_name"] = graph_name
        return get_daemon_handle(dlc_path, runtime=runtime, perf_profile=perf_profile, **options).client
    except Exception as exc:
        print(f"[SnpeDaemon] unavailable ({exc})")
        return None
