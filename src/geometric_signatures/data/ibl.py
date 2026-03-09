"""IBL (International Brain Laboratory) Neuropixels data loader.

Loads electrophysiology recordings from the IBL public database via
the ONE API. Requires ``ONE-api`` package::

    pip install ONE-api

Authentication::

    from one.api import ONE
    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        password='international',
    )

Typical workflow::

    from geometric_signatures.data.ibl import load_ibl_session
    data = load_ibl_session('session_eid', cache_dir=Path('data/ibl'))
    # data is a NeuralPopulationData ready for analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..population import NeuralPopulationData

logger = logging.getLogger(__name__)

try:
    from one.api import ONE

    ONE_AVAILABLE = True
except ImportError:
    ONE_AVAILABLE = False


def _check_one_available() -> None:
    """Raise ImportError if ONE API is not installed."""
    if not ONE_AVAILABLE:
        raise ImportError(
            "IBL data loading requires ONE-api. Install with: "
            "pip install ONE-api"
        )


def load_ibl_session(
    session_eid: str,
    cache_dir: Path,
    brain_region: str | None = None,
    bin_size: float = 0.01,
    align_event: str = "stimOn_times",
    window: tuple[float, float] = (-0.5, 1.5),
    normalize: str = "zscore",
) -> NeuralPopulationData:
    """Load an IBL Neuropixels session as NeuralPopulationData.

    Downloads spike times and cluster information, bins spikes, aligns
    to trial events, and normalizes.

    Args:
        session_eid: IBL session identifier (UUID string).
        cache_dir: Local cache directory for downloaded data.
        brain_region: Filter clusters by Allen atlas region (e.g., "VISp").
            None = use all clusters.
        bin_size: Spike binning interval in seconds (default 10ms).
        align_event: Trial event to align to (default "stimOn_times").
        window: Time window around event in seconds (default -0.5 to +1.5).
        normalize: Normalization method ("zscore", "max", "range").

    Returns:
        NeuralPopulationData with source="ibl".

    Raises:
        ImportError: If ONE-api is not installed.
        ValueError: If session data cannot be loaded.
    """
    _check_one_available()

    from .neural_preprocessing import align_trials, bin_spikes, normalize_population

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=str(cache_dir),
        silent=True,
    )

    logger.info("Loading IBL session: %s", session_eid)

    # Load spike data
    spike_times, spike_clusters = one.load_datasets(
        session_eid,
        datasets=["spikes.times", "spikes.clusters"],
    )

    # Load cluster info for brain region filtering
    if brain_region is not None:
        cluster_regions = one.load_dataset(
            session_eid, "clusters.acronym"
        )
        valid_clusters = np.where(
            np.array(cluster_regions) == brain_region
        )[0]
        mask = np.isin(spike_clusters, valid_clusters)
        spike_times = spike_times[mask]
        spike_clusters = spike_clusters[mask]
        logger.info(
            "Filtered to %d clusters in %s", len(valid_clusters), brain_region
        )

    # Bin spikes
    rates, bin_centers, cluster_ids = bin_spikes(
        spike_times, spike_clusters, bin_size
    )

    # Load trial event times
    trials = one.load_object(session_eid, "trials")
    event_times = getattr(trials, align_event, None)
    if event_times is None:
        raise ValueError(
            f"Event '{align_event}' not found in trials object. "
            f"Available: {[a for a in dir(trials) if not a.startswith('_')]}"
        )

    # Remove NaN event times
    valid_trials = ~np.isnan(event_times)
    event_times = event_times[valid_trials]

    # Trial labels from IBL trial types
    trial_labels = _extract_ibl_trial_labels(trials, valid_trials)

    # Align to trials
    aligned, trial_time = align_trials(
        rates, bin_centers, event_times, window
    )

    # Normalize
    if normalize != "none":
        aligned = normalize_population(aligned, method=normalize)

    # Build unit labels
    unit_labels = tuple(f"cluster_{int(c)}" for c in cluster_ids)

    return NeuralPopulationData(
        activity=aligned,
        trial_labels=trial_labels,
        time_axis=trial_time,
        unit_labels=unit_labels,
        source="ibl",
        metadata={
            "session_eid": session_eid,
            "brain_region": brain_region or "all",
            "bin_size": bin_size,
            "align_event": align_event,
            "window": list(window),
            "n_clusters": len(cluster_ids),
        },
    )


def _extract_ibl_trial_labels(
    trials: Any,
    valid_mask: np.ndarray,
) -> tuple[str, ...]:
    """Extract trial labels from IBL trials object."""
    # Try common IBL trial type fields
    if hasattr(trials, "contrastLeft") and hasattr(trials, "contrastRight"):
        left = np.array(trials.contrastLeft)[valid_mask]
        right = np.array(trials.contrastRight)[valid_mask]
        labels = []
        for l_val, r_val in zip(left, right):
            if np.isnan(l_val) and not np.isnan(r_val):
                labels.append("right_stim")
            elif not np.isnan(l_val) and np.isnan(r_val):
                labels.append("left_stim")
            elif not np.isnan(l_val) and not np.isnan(r_val):
                labels.append("both_stim")
            else:
                labels.append("no_stim")
        return tuple(labels)

    # Fallback: generic labels
    n_trials = int(valid_mask.sum())
    return tuple(f"trial_{i}" for i in range(n_trials))


def list_ibl_sessions(
    brain_region: str | None = None,
    cache_dir: Path | None = None,
) -> list[str]:
    """List available IBL sessions.

    Args:
        brain_region: Filter by brain region (e.g., "VISp").
        cache_dir: Local cache directory.

    Returns:
        List of session EIDs.
    """
    _check_one_available()

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=str(cache_dir) if cache_dir else None,
        silent=True,
    )

    kwargs: dict[str, Any] = {"dataset": "spikes.times"}
    if brain_region is not None:
        kwargs["atlas_acronym"] = brain_region

    sessions = one.search(**kwargs)
    return [str(s) for s in sessions]
