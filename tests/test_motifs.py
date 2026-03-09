from geometric_signatures.motifs import MotifSwitches, build_single_ablation_variants


def test_single_ablation_variants_cover_all_motifs() -> None:
    base = MotifSwitches(True, True, True, True)
    variants = build_single_ablation_variants(base)

    assert "complete" in variants
    assert len(variants) == 5
    assert variants["ablate_normalization_gain_modulation"].normalization_gain_modulation is False
    assert variants["ablate_attractor_dynamics"].attractor_dynamics is False
    assert variants["ablate_selective_gating"].selective_gating is False
    assert variants["ablate_expansion_recoding"].expansion_recoding is False
