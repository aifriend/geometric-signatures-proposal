"""Biological data loaders and neural preprocessing.

Provides loaders for:
- **IBL** (International Brain Laboratory): Neuropixels recordings via ONE API.
- **Allen Brain Observatory**: Visual Behavior Ophys experiments via AllenSDK.

Both loaders output ``NeuralPopulationData`` — the same contract used by
RNN models — so the analysis pipeline works identically on biological data.

Also provides neural preprocessing utilities:
- Spike binning (electrophysiology → rate)
- ΔF/F computation (calcium imaging)
- Trial-aligned extraction
- Population normalization
"""

from .neural_preprocessing import (
    align_trials,
    bin_spikes,
    compute_delta_f_over_f,
    normalize_population,
)

__all__ = [
    "bin_spikes",
    "compute_delta_f_over_f",
    "align_trials",
    "normalize_population",
]
