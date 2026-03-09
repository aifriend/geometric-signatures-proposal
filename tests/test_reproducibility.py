"""Tests for the reproducibility module.

Validates:
- Deterministic seeding produces identical sequences.
- Environment capture returns expected keys.
- Functions degrade gracefully without PyTorch.
"""

from __future__ import annotations

import random

import numpy as np

import pytest

from geometric_signatures.reproducibility import (
    capture_environment,
    resolve_device,
    set_all_seeds,
)


# ── Seed determinism ───────────────────────────────────────────


class TestSetAllSeeds:
    """Verify that set_all_seeds produces deterministic sequences."""

    def test_numpy_determinism(self) -> None:
        """Same seed → identical numpy random sequences."""
        set_all_seeds(42)
        seq1 = np.random.rand(10)

        set_all_seeds(42)
        seq2 = np.random.rand(10)

        np.testing.assert_array_equal(seq1, seq2)

    def test_python_random_determinism(self) -> None:
        """Same seed → identical Python random sequences."""
        set_all_seeds(42)
        seq1 = [random.random() for _ in range(10)]

        set_all_seeds(42)
        seq2 = [random.random() for _ in range(10)]

        assert seq1 == seq2

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Different seeds → different sequences."""
        set_all_seeds(42)
        seq1 = np.random.rand(10)

        set_all_seeds(99)
        seq2 = np.random.rand(10)

        assert not np.array_equal(seq1, seq2)

    def test_torch_determinism_if_available(self) -> None:
        """If torch is installed, verify deterministic tensor generation."""
        try:
            import torch
        except ImportError:
            return  # Skip if torch not installed

        set_all_seeds(42)
        t1 = torch.randn(10)

        set_all_seeds(42)
        t2 = torch.randn(10)

        assert torch.equal(t1, t2)


# ── Environment capture ────────────────────────────────────────


class TestCaptureEnvironment:
    """Verify environment capture returns expected fields."""

    def test_required_keys_present(self) -> None:
        env = capture_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "numpy_version" in env
        assert "git_hash" in env
        assert "git_dirty" in env
        assert "cwd" in env

    def test_python_version_format(self) -> None:
        env = capture_environment()
        parts = env["python_version"].split(".")
        assert len(parts) == 3  # major.minor.patch
        assert all(part.isdigit() for part in parts)

    def test_numpy_version_populated(self) -> None:
        env = capture_environment()
        # Should be a non-empty version string like "1.24.0"
        assert len(env["numpy_version"]) > 0
        assert "." in env["numpy_version"]

    def test_torch_version_present(self) -> None:
        """torch_version should be present (either version string or 'not_installed')."""
        env = capture_environment()
        assert "torch_version" in env
        # Either a version string or "not_installed"
        assert len(env["torch_version"]) > 0

    def test_git_hash_present(self) -> None:
        """git_hash should be present (either hash or 'unknown')."""
        env = capture_environment()
        assert env["git_hash"] in ("unknown",) or len(env["git_hash"]) >= 7

    def test_mps_available_key(self) -> None:
        """mps_available should be present when torch is installed."""
        env = capture_environment()
        if env["torch_version"] != "not_installed":
            assert "mps_available" in env
            assert env["mps_available"] in ("True", "False")


# ── Device resolution ─────────────────────────────────────────


class TestResolveDevice:
    """Verify resolve_device validates and auto-detects devices."""

    def test_cpu_always_valid(self) -> None:
        """CPU device is always available."""
        assert resolve_device("cpu") == "cpu"

    def test_auto_returns_valid_device(self) -> None:
        """Auto selects best available device."""
        result = resolve_device("auto")
        assert result in ("cpu", "cuda", "mps")

    def test_auto_is_default(self) -> None:
        """Default argument is 'auto'."""
        result = resolve_device()
        assert result in ("cpu", "cuda", "mps")

    def test_case_insensitive(self) -> None:
        """Device strings are case-insensitive."""
        assert resolve_device("CPU") == "cpu"
        assert resolve_device("Cpu") == "cpu"

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert resolve_device("  cpu  ") == "cpu"

    def test_unknown_device_raises(self) -> None:
        """Unknown device string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_device("tpu")

    def test_mps_on_this_machine(self) -> None:
        """On Apple Silicon, MPS should be available."""
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert resolve_device("mps") == "mps"
        else:
            with pytest.raises(ValueError, match="MPS is not available"):
                resolve_device("mps")

    def test_cuda_validation(self) -> None:
        """CUDA raises ValueError if not available."""
        import torch

        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA is not available"):
                resolve_device("cuda")

    def test_auto_prefers_accelerator(self) -> None:
        """Auto should prefer cuda or mps over cpu when available."""
        import torch

        result = resolve_device("auto")
        if torch.cuda.is_available():
            assert result == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert result == "mps"
        else:
            assert result == "cpu"
