"""Background resource sampling; missing instruments remain explicitly unknown.

Adapted from legacy resource_sampler. Thermal raw units are NOT guessed from
magnitude. RSS covers only this process, not shared SNPE/Cloud daemons.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from time import monotonic_ns, time_ns

from .trace import JsonlWriter


_GPU_FIELDS = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw")


def sample_gpu() -> dict | None:
    """One nvidia-smi reading for GPU0; None when no NVIDIA tool/GPU (never 0)."""
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        line = subprocess.run(["nvidia-smi", "--id=0", f"--query-gpu={','.join(_GPU_FIELDS)}",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=2, check=True).stdout.strip().splitlines()[0]
        values = [float(item) if item.strip() not in {"[N/A]", "N/A"} else None for item in line.split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"error": "nvidia-smi query failed"}
    util, used, total, temp, power = values
    return {"utilization_pct": util, "memory_used_mib": used, "memory_total_mib": total,
            "temperature_c": temp, "power_w": power}


class ResourceSampler:
    def __init__(self, path: Path, node_id: str, interval_s: float = 1.0) -> None:
        if interval_s < 0.1:
            raise ValueError("resource interval must be >=0.1s")
        self.node_id = node_id
        self.interval_s = interval_s
        self.writer = JsonlWriter(path, capacity=64)
        self._stop = threading.Event()
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="bench-resources")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                process.cpu_percent(None)
                psutil.cpu_percent(None)
            except ImportError:
                psutil = process = None
            while not self._stop.wait(self.interval_s):
                row = {"schema_version": 1, "node_id": self.node_id,
                       "mono_ns": monotonic_ns(), "utc_ns": time_ns(),
                       "process_rss_bytes": None, "process_cpu_percent": None,
                       "host_cpu_percent": None, "memory_available_bytes": None,
                       "host_net_sent_bytes": None, "host_net_recv_bytes": None,
                       "gpu_utilization": None, "npu_utilization": None,
                       "thermal_raw": {}, "temperature_c": None,
                       "scope": "process-only RSS; host-wide CPU/network",
                       "sampler_available": psutil is not None}
                if psutil is not None:
                    net = psutil.net_io_counters()
                    row.update(process_rss_bytes=process.memory_info().rss,
                               process_cpu_percent=process.cpu_percent(None),
                               host_cpu_percent=psutil.cpu_percent(None),
                               memory_available_bytes=psutil.virtual_memory().available,
                               host_net_sent_bytes=net.bytes_sent, host_net_recv_bytes=net.bytes_recv)
                row["gpu"] = sample_gpu()
                if row["gpu"] and row["gpu"].get("utilization_pct") is not None:
                    row["gpu_utilization"] = row["gpu"]["utilization_pct"]
                for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                    try:
                        row["thermal_raw"][str(path)] = float(path.read_text().strip())
                    except (OSError, ValueError):
                        pass
                self.writer.write(row)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("resource sampler drain timeout")
        self.writer.close()
        if self.error:
            raise RuntimeError(self.error)
