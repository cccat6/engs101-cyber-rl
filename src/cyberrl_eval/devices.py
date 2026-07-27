"""PyTorch device selection shared by the local and cloud runners."""

from __future__ import annotations

import random

import numpy as np
import torch


_ALIASES = {
    "metal": "mps",
}


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and bool(xpu.is_available())


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and bool(backend.is_available())


def resolve_device(requested: str) -> str:
    """Resolve and validate a student-facing PyTorch device string.

    ``auto`` prefers CUDA, then Intel XPU, then Apple Metal/MPS, and finally
    CPU. ``metal`` is accepted as a friendly alias for PyTorch's ``mps``.
    The returned value is suitable for ``torch.device(...)``.
    """

    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device must be a non-empty string")
    normalized = requested.strip().lower()
    normalized = _ALIASES.get(normalized, normalized)

    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif _xpu_available():
            normalized = "xpu"
        elif _mps_available():
            normalized = "mps"
        else:
            normalized = "cpu"

    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Unsupported PyTorch device {requested!r}") from error

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
    elif device.type == "xpu":
        if not _xpu_available():
            raise RuntimeError("Intel XPU was requested but is not available")
        device_count = getattr(torch.xpu, "device_count", lambda: 1)()
        if device.index is not None and device.index >= device_count:
            raise RuntimeError(
                f"XPU device {device.index} was requested, but only "
                f"{device_count} device(s) are visible"
            )
    elif device.type == "mps":
        if not _mps_available():
            raise RuntimeError("Apple Metal/MPS was requested but is not available")
        if device.index not in (None, 0):
            raise RuntimeError("Apple Metal/MPS does not support multiple device indices")
    elif device.type != "cpu":
        raise ValueError(
            f"Device type {device.type!r} is not supported; choose "
            "auto, cpu, cuda, xpu, mps, or metal"
        )

    return str(device)


def seed_everything(seed: int) -> None:
    """Seed the ordinary RNGs without forcing slow deterministic kernels."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _xpu_available():
        torch.xpu.manual_seed_all(seed)
