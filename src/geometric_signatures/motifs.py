from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MotifSwitches:
    normalization_gain_modulation: bool
    attractor_dynamics: bool
    selective_gating: bool
    expansion_recoding: bool

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Single source of truth for motif field names."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MotifSwitches":
        expected = cls.field_names()
        missing = [key for key in expected if key not in data]
        if missing:
            raise ValueError(f"Missing motif keys: {', '.join(missing)}")

        return cls(**{key: bool(data[key]) for key in expected})


def build_single_ablation_variants(base: MotifSwitches) -> dict[str, MotifSwitches]:
    variants: dict[str, MotifSwitches] = {"complete": base}
    for key in MotifSwitches.field_names():
        values = asdict(base)
        values[key] = False
        variants[f"ablate_{key}"] = MotifSwitches(**values)
    return variants
