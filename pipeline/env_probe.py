from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _run_command(command: list[str], timeout_seconds: int = 10) -> str | None:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception:
        return None
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        return None if not output else output
    return output or None


def probe_nvidia_smi() -> str | None:
    return _run_command(["nvidia-smi"], timeout_seconds=8)


def probe_torch() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        return {
            "torch_version": str(getattr(torch, "__version__", None)),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception:
        return {
            "torch_version": None,
            "cuda_available": False,
        }


def probe_environment() -> dict[str, Any]:
    torch_info = probe_torch()
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": torch_info["cuda_available"],
        "torch_version": torch_info["torch_version"],
        "nvidia_smi": probe_nvidia_smi(),
    }

