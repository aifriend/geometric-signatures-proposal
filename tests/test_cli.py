"""Tests for CLI entry points.

Tests cover argument parsing, --help smoke tests, and basic error handling.
Full integration tests (train/analyze end-to-end) are covered by
test_pipeline/ — here we focus on CLI plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geometric_signatures.cli import build_parser, main


class TestBuildParser:
    """Tests for argument parser construction."""

    def test_parser_creates(self) -> None:
        """Parser is created without errors."""
        parser = build_parser()
        assert parser is not None

    def test_has_all_subcommands(self) -> None:
        """All four subcommands are registered."""
        parser = build_parser()
        # Parse known subcommands — shouldn't raise
        args = parser.parse_args(["train", "config.yaml"])
        assert args.command == "train"

        args = parser.parse_args(["analyze", "config.yaml"])
        assert args.command == "analyze"

        args = parser.parse_args(["compare", "dir_a", "dir_b"])
        assert args.command == "compare"

        args = parser.parse_args(["status", "runs/"])
        assert args.command == "status"


class TestTrainArgs:
    """Tests for train subcommand argument parsing."""

    def test_default_values(self) -> None:
        """Train subcommand has correct defaults."""
        parser = build_parser()
        args = parser.parse_args(["train", "config.yaml"])

        assert args.config == Path("config.yaml")
        assert args.output_dir == Path("runs")
        assert args.device == "auto"
        assert args.variants is None
        assert args.log_level == "INFO"

    def test_custom_values(self) -> None:
        """Train subcommand accepts custom values."""
        parser = build_parser()
        args = parser.parse_args([
            "train", "my_config.yaml",
            "--output-dir", "/tmp/output",
            "--device", "cuda",
            "--variants", "complete", "ablate_gating",
            "--log-level", "DEBUG",
        ])

        assert args.config == Path("my_config.yaml")
        assert args.output_dir == Path("/tmp/output")
        assert args.device == "cuda"
        assert args.variants == ["complete", "ablate_gating"]
        assert args.log_level == "DEBUG"


class TestAnalyzeArgs:
    """Tests for analyze subcommand argument parsing."""

    def test_default_values(self) -> None:
        """Analyze subcommand has correct defaults."""
        parser = build_parser()
        args = parser.parse_args(["analyze", "config.yaml"])

        assert args.config == Path("config.yaml")
        assert args.output_dir == Path("runs")
        assert args.methods is None
        assert args.variants is None
        assert args.with_stats is False

    def test_with_methods(self) -> None:
        """Methods are parsed as comma-separated string."""
        parser = build_parser()
        args = parser.parse_args([
            "analyze", "config.yaml",
            "--methods", "cka,rsa,population_geometry",
        ])
        assert args.methods == "cka,rsa,population_geometry"

    def test_with_stats_flag(self) -> None:
        """--with-stats flag sets with_stats to True."""
        parser = build_parser()
        args = parser.parse_args(["analyze", "config.yaml", "--with-stats"])
        assert args.with_stats is True


class TestCompareArgs:
    """Tests for compare subcommand argument parsing."""

    def test_required_dirs(self) -> None:
        """Compare requires two positional directory arguments."""
        parser = build_parser()
        args = parser.parse_args(["compare", "runs/rnn", "runs/bio"])

        assert args.dir_a == Path("runs/rnn")
        assert args.dir_b == Path("runs/bio")

    def test_default_values(self) -> None:
        """Compare has correct defaults."""
        parser = build_parser()
        args = parser.parse_args(["compare", "a", "b"])

        assert args.output is None
        assert args.methods is None
        assert args.n_permutations == 1000
        assert args.alpha == 0.05

    def test_custom_values(self) -> None:
        """Compare accepts custom parameters."""
        parser = build_parser()
        args = parser.parse_args([
            "compare", "dir_a", "dir_b",
            "--output", "figures/",
            "--methods", "cka,rsa",
            "--n-permutations", "5000",
            "--alpha", "0.01",
        ])

        assert args.output == Path("figures/")
        assert args.methods == "cka,rsa"
        assert args.n_permutations == 5000
        assert args.alpha == 0.01


class TestStatusArgs:
    """Tests for status subcommand argument parsing."""

    def test_required_dir(self) -> None:
        """Status requires catalog directory."""
        parser = build_parser()
        args = parser.parse_args(["status", "runs/"])
        assert args.catalog_dir == Path("runs/")

    def test_filters(self) -> None:
        """Status accepts variant and status filters."""
        parser = build_parser()
        args = parser.parse_args([
            "status", "runs/",
            "--variant", "complete",
            "--status", "completed",
        ])
        assert args.variant == "complete"
        assert args.filter_status == "completed"


class TestMainEntry:
    """Tests for the main() entry point."""

    def test_no_command_returns_zero(self) -> None:
        """Calling with no command prints help and returns 0."""
        exit_code = main([])
        assert exit_code == 0

    def test_help_flag(self) -> None:
        """--help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_version_flag(self) -> None:
        """--version raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_train_help(self) -> None:
        """train --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--help"])
        assert exc_info.value.code == 0

    def test_analyze_help(self) -> None:
        """analyze --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", "--help"])
        assert exc_info.value.code == 0

    def test_compare_help(self) -> None:
        """compare --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["compare", "--help"])
        assert exc_info.value.code == 0

    def test_status_help(self) -> None:
        """status --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["status", "--help"])
        assert exc_info.value.code == 0

    def test_train_missing_config(self) -> None:
        """Train with nonexistent config returns error."""
        exit_code = main(["train", "nonexistent.yaml"])
        assert exit_code == 1

    def test_status_missing_catalog(self, tmp_path: Path) -> None:
        """Status with missing catalog returns error."""
        exit_code = main(["status", str(tmp_path)])
        assert exit_code == 1

    def test_status_with_catalog(self, tmp_path: Path) -> None:
        """Status with empty catalog returns zero."""
        from geometric_signatures.tracking import ExperimentCatalog

        catalog = ExperimentCatalog(tmp_path / "experiment_catalog.db")
        exit_code = main(["status", str(tmp_path)])
        assert exit_code == 0

    def test_compare_missing_dirs(self, tmp_path: Path) -> None:
        """Compare with empty directories returns error."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        exit_code = main(["compare", str(dir_a), str(dir_b)])
        assert exit_code == 1
