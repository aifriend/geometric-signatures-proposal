"""Deterministic seeding, device resolution, and environment capture.

Every training run and analysis pipeline must produce identical results
given the same seed. This module provides:

- ``set_all_seeds``: Seed Python, NumPy, and optionally PyTorch + CUDA.
- ``enable_deterministic_mode``: Force deterministic algorithms in PyTorch.
- ``resolve_device``: Validate and auto-detect the best available device
  (cpu, cuda, mps).
- ``capture_environment``: Record git hash, package versions, and platform
  info for inclusion in run manifests.

PyTorch is optional — these utilities degrade gracefully when torch is not
installed (Steps 1-2 of the plan require only numpy).
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Set deterministic seeds for Python random, NumPy, and PyTorch.

    If PyTorch is installed, also seeds torch and CUDA. Safe to call
    when torch is not available (e.g., in analysis-only environments).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional: PyTorch seeding (only if installed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str = "auto") -> str:
    """Validate and resolve the training device string.

    Accepts ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.

    - ``"auto"``: Picks the best available accelerator —
      CUDA if available, then MPS (Apple Silicon), else CPU.
    - ``"cuda"`` / ``"cuda:N"``: Validated against ``torch.cuda.is_available()``.
    - ``"mps"``: Validated against ``torch.backends.mps.is_available()``.
    - ``"cpu"``: Always valid.

    Args:
        device: Requested device string. Defaults to ``"auto"``.

    Returns:
        Resolved device string ready for ``model.to(device)``.

    Raises:
        ImportError: If torch is not installed.
        ValueError: If the requested device is not available.
    """
    import torch

    device = device.strip().lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cpu":
        return "cpu"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{device}' but CUDA is not available. "
                "Use --device cpu or --device auto."
            )
        return device

    if device == "mps":
        if not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ValueError(
                "Requested device 'mps' but MPS is not available. "
                "Requires macOS 12.3+ with Apple Silicon and PyTorch >= 1.12. "
                "Use --device cpu or --device auto."
            )
        return "mps"

    raise ValueError(
        f"Unknown device: '{device}'. "
        "Valid options: auto, cpu, cuda, cuda:N, mps."
    )


def enable_deterministic_mode() -> None:
    """Force deterministic algorithms in PyTorch.

    This may reduce performance but guarantees bitwise reproducibility
    across runs on the same hardware. No-op if torch is not installed.

    Note:
        Some operations have no deterministic implementation and will raise
        RuntimeError. Use ``torch.use_deterministic_algorithms(mode=True,
        warn_only=True)`` as a softer alternative during development.
    """
    try:
        import torch

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def capture_environment() -> dict[str, str]:
    """Record runtime environment for inclusion in run manifests.

    Returns a dict with:
        - ``python_version``: e.g. "3.10.12"
        - ``platform``: e.g. "macOS-14.0-arm64"
        - ``numpy_version``: installed numpy version
        - ``torch_version``: installed torch version (or "not_installed")
        - ``git_hash``: current HEAD short hash (or "unknown")
        - ``git_dirty``: "true" if working tree has uncommitted changes
        - ``cwd``: current working directory
    """
    env: dict[str, str] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }

    # NumPy version
    env["numpy_version"] = np.__version__

    # PyTorch version (optional)
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "unknown"
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        env["mps_available"] = str(mps_available)
    except ImportError:
        env["torch_version"] = "not_installed"

    # Git hash
    env["git_hash"] = _get_git_hash()
    env["git_dirty"] = _is_git_dirty()

    # Key package versions
    for pkg in ("PyYAML", "scipy", "scikit-learn", "ripser", "rsatoolbox"):
        try:
            env[f"{pkg}_version"] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass  # Only record installed packages

    return env


def _get_git_hash() -> str:
    """Get the current git short hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _is_git_dirty() -> str:
    """Check if the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return str(bool(result.stdout.strip())).lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"
