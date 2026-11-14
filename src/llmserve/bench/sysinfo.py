"""Hardware/environment facts recorded with every benchmark result."""

from __future__ import annotations

import platform
import shutil
import subprocess


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def git_commit() -> str | None:
    sha = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return f"{sha}{'-dirty' if dirty else ''}" if sha else None


def gpu_memory_mib() -> list[dict] | None:
    """Per-GPU memory in use via nvidia-smi, or None when there is no NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    gpus = []
    for line in out.splitlines():
        idx, name, used, total = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "used_mib": float(used), "total_mib": float(total)})
    return gpus


def hardware() -> dict:
    cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    gpus = gpu_memory_mib()
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu,
        "gpus": [g["name"] for g in gpus] if gpus else [],
        "python": platform.python_version(),
    }
