"""QAIRT / SNPE environment helpers (edge-246 pin 2.43)."""

from __future__ import annotations

import os
import sys


def qairt_root() -> str:
    return os.environ.get("QAIRT_ROOT", os.path.expanduser("~/sigm/qairt/2.43.0.260128"))


def qairt_abi() -> str:
    # Edge246 (Ubuntu 24.04) uses the OE gcc11.2 libs; Ubuntu 20.04 Edges set aarch64-ubuntu-gcc9.4.
    return os.environ.get("QAIRT_ABI", "aarch64-oe-linux-gcc11.2")


def setup_qairt_env() -> dict[str, str]:
    """Return env dict with LD_LIBRARY_PATH and PYTHONPATH for SNPE Python."""
    root = qairt_root()
    abi = qairt_abi()
    env = os.environ.copy()
    lib = f"{root}/lib/{abi}"
    py = f"{root}/lib/python"
    env["QAIRT_ROOT"] = root
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if os.path.isdir(py):
        env["PYTHONPATH"] = f"{py}:{env.get('PYTHONPATH', '')}"
    return env


def setup_yolo11s_candidate_env() -> dict[str, str]:
    """Candidate child only: mirror the successful split-v8 video runtime paths."""
    env = setup_qairt_env()
    root = env["QAIRT_ROOT"]
    target = f"{root}/lib/{qairt_abi()}"
    env["LD_LIBRARY_PATH"] = f"{target}:{sys.prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    env["ADSP_LIBRARY_PATH"] = ";".join((f"{root}/lib/hexagon-v68/unsigned", target,
                                         "/usr/lib/rfsa/adsp", "/dsp", "/opt"))
    return env


def apply_qairt_to_process() -> None:
    """Mutate current process env/sys.path so `import snpe` works."""
    env = setup_qairt_env()
    os.environ.update(env)
    py = f"{qairt_root()}/lib/python"
    if os.path.isdir(py) and py not in sys.path:
        sys.path.insert(0, py)


def snpe_persistent_enabled() -> bool:
    raw = os.environ.get("SNPE_PERSISTENT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
