"""Tests for figure style presets."""

from __future__ import annotations

import matplotlib as mpl
import pytest

from geometric_signatures.figures.style import STYLE_PRESETS, apply_style


class TestStylePresets:
    """Tests for style preset dictionary."""

    def test_all_presets_exist(self) -> None:
        """Paper, poster, and presentation presets are defined."""
        assert "paper" in STYLE_PRESETS
        assert "poster" in STYLE_PRESETS
        assert "presentation" in STYLE_PRESETS

    def test_presets_are_dicts(self) -> None:
        """Each preset maps strings to values."""
        for name, preset in STYLE_PRESETS.items():
            assert isinstance(preset, dict), f"{name} is not a dict"
            for key in preset:
                assert isinstance(key, str), f"Non-string key in {name}: {key}"

    def test_paper_preset_has_small_fonts(self) -> None:
        """Paper preset uses small fonts suitable for journal columns."""
        paper = STYLE_PRESETS["paper"]
        assert paper["font.size"] <= 8
        assert paper["figure.dpi"] >= 300

    def test_poster_preset_has_large_fonts(self) -> None:
        """Poster preset uses large fonts for readability at distance."""
        poster = STYLE_PRESETS["poster"]
        assert poster["font.size"] >= 16

    def test_presentation_preset_has_dark_background(self) -> None:
        """Presentation preset uses dark background colors."""
        pres = STYLE_PRESETS["presentation"]
        assert pres["figure.facecolor"] == "#2E2E2E"
        assert pres["axes.facecolor"] == "#2E2E2E"

    def test_all_presets_hide_top_right_spines(self) -> None:
        """All presets remove top and right spines for clean plots."""
        for name, preset in STYLE_PRESETS.items():
            assert preset.get("axes.spines.top") is False, (
                f"{name} has top spine enabled"
            )
            assert preset.get("axes.spines.right") is False, (
                f"{name} has right spine enabled"
            )

    def test_all_presets_have_color_cycle(self) -> None:
        """All presets define a color cycle with 8 colors."""
        for name, preset in STYLE_PRESETS.items():
            cycle = preset.get("axes.prop_cycle")
            assert cycle is not None, f"{name} missing color cycle"
            # Cycler stores colors as list of dicts
            colors = list(cycle)
            assert len(colors) == 8, f"{name} has {len(colors)} colors, expected 8"


class TestApplyStyle:
    """Tests for apply_style function."""

    def teardown_method(self) -> None:
        """Reset rcParams after each test."""
        mpl.rcdefaults()

    def test_apply_paper(self) -> None:
        """Applying paper style sets font family to serif."""
        apply_style("paper")
        assert mpl.rcParams["font.family"] == ["serif"]

    def test_apply_poster(self) -> None:
        """Applying poster style sets large font size."""
        apply_style("poster")
        assert mpl.rcParams["font.size"] >= 16

    def test_apply_presentation(self) -> None:
        """Applying presentation style sets dark background."""
        apply_style("presentation")
        assert mpl.rcParams["figure.facecolor"] == "#2E2E2E"

    def test_unknown_preset_raises(self) -> None:
        """Unknown preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown style preset"):
            apply_style("nonexistent")

    def test_apply_resets_before_applying(self) -> None:
        """Applying a style first resets to defaults, then applies."""
        # Apply presentation (dark bg), then paper — should NOT inherit dark bg
        apply_style("presentation")
        apply_style("paper")

        # Paper has no dark background setting — rcParams should reflect paper
        assert mpl.rcParams["font.family"] == ["serif"]

    def test_apply_all_presets_without_error(self) -> None:
        """All preset names can be applied without exceptions."""
        for name in STYLE_PRESETS:
            apply_style(name)
