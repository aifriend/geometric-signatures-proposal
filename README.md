<h1 align="center">Geometric Signatures of Computational Motifs</h1>

<p align="center">
  <em>PhD codebase — discovering reusable computational motifs in neural population dynamics, and testing whether they survive the jump from constrained RNNs to real cortex.</em>
</p>

<p align="center">
  <img alt="Python"    src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white">
  <img alt="PyTorch"   src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
  <img alt="uv"        src="https://img.shields.io/badge/uv-DE5FE9?style=flat">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white">
  <a href="https://github.com/aifriend/geometric-signatures-proposal/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aifriend/geometric-signatures-proposal/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License"   src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Status"    src="https://img.shields.io/badge/phase-3%20(biological%20validation)-blue.svg">
</p>

<!--
TODO: Drop a single hero panel from Phase 2 results here.
Best candidate: a 2x2 grid showing (top) RDM heatmaps for 4 motifs and (bottom)
a t-SNE / persistent-homology summary. Save as docs/hero.png.
-->
<p align="center">
  <img src="docs/hero.png" alt="Phase 2 motif signatures across 50 seeds" width="820">
</p>

---

## The question

Cortical circuits seem to reuse a small alphabet of **computational motifs** — normalization and gain modulation, attractor dynamics, selective gating, expansion recoding. We have good math for each motif in isolation. What we don't have is a way to **detect them from population activity alone**, the way an experimentalist could detect them from a recording session.

This thesis asks: do these motifs leave **geometric signatures** in the trajectories of neural population activity, signatures stable enough to identify the motif from data without knowing the underlying circuit?

If yes, the same diagnostic should work on artificial RNNs trained on a task *and* on real cortical recordings doing the same kind of task. That's the bet.

## The plan, in three phases

**Phase 1 — Theory and design.** Define four canonical motifs (normalization and gain modulation, attractor dynamics, selective gating, expansion recoding), specify how each should distort population geometry, predict the signature each leaves under three quantitative tools.

**Phase 2 — Validation in silico.** ✅ **Done.** Train constrained RNNs to solve task variants that induce each motif. Across **50 seeds × 4 motif conditions = 200 trained models**, measure the geometric signatures and confirm they cluster by motif, not by random init.

**Phase 3 — Validation in vivo.** 🚧 **Current focus.** Apply the same pipeline to two public datasets:
- **International Brain Laboratory** (IBL) decision-making task — population activity from multiple cortical regions
- **Allen Brain Observatory** — visual cortex under structured stimuli

If the in-silico motif signatures map onto real population data, we have a tool. If they don't, we learn where the abstraction breaks.

## How motifs are measured

Three complementary geometric tools, applied to the same trial-by-trial population activity:

| Tool | What it captures | Sensitive to |
|---|---|---|
| **RSA / CKA** | Pairwise similarity of population states across conditions | Coarse coding geometry |
| **Persistent homology** | Topological structure (loops, voids) in the trajectory manifold | Attractor topology |
| **MARBLE** | Local geometric invariants over the trajectory | Motif-level dynamics |

A motif "lives" if its signature is consistent across seeds *within* a motif and distinct *between* motifs. Phase 2 confirmed all four motifs separate cleanly on at least two of the three tools.

## Project structure

```
.
├── src/geometric_signatures/
│   ├── cli.py              # `geometric-signatures` entry point (train, analyze, compare, status)
│   ├── motifs.py           # The four motif switches and single-ablation variants
│   ├── tasks.py            # Task battery definitions
│   ├── config.py           # Experiment YAML loading
│   ├── population.py       # Population activity container
│   ├── models/             # Constrained RNN architectures
│   ├── training/           # Trainer and checkpoints
│   ├── analysis/           # Similarity (CKA/RSA), geometry, topology and MARBLE methods
│   ├── statistics/         # Bootstrap, permutation tests, aggregation, multiple-comparison correction
│   ├── comparison/         # Cross-system (RNN vs. biological) comparison
│   ├── data/               # IBL and Allen loaders, neural preprocessing
│   ├── pipeline/           # Train / analyze / statistics stages and runner
│   ├── figures/            # Plotting and paper style
│   ├── tracking.py         # Experiment catalog
│   └── reproducibility.py  # Seeding
├── config/                 # Experiment YAML configs (baseline and ablation template)
├── scripts/                # Phase 2 figure script and Streamlit dashboard
├── figures/phase2/         # Phase 2 figures and comparison data
├── tests/
├── pyproject.toml          # uv / PEP 621
└── README.md
```

## Getting started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (or any modern Python env manager)
- A GPU is strongly recommended for training (any 12GB+ NVIDIA card works); analysis runs on CPU

### Installation

```bash
git clone https://github.com/aifriend/geometric-signatures-proposal.git
cd geometric-signatures-proposal
uv sync --extra dev --extra train --extra analysis --extra figures --extra dashboard
```

Training needs the `train` extra (PyTorch), the analysis methods need `analysis`, figures need `figures`, and the dashboard needs `dashboard`.
MARBLE is installed separately; see the note in `pyproject.toml`.

### Reproduce Phase 2

```bash
# Train the complete model and every single-motif ablation (multi-seed)
uv run geometric-signatures train config/experiment.baseline.yaml --output-dir runs/

# Analyze the trained models
uv run geometric-signatures analyze config/experiment.baseline.yaml --output-dir runs/ --methods cka,rsa --with-stats

# Check which runs completed
uv run geometric-signatures status runs/

# Launch the dashboard
uv run streamlit run scripts/streamlit_dashboard.py
```

### Scripts

Helper scripts for long training runs live in `scripts/`. The Google Cloud ones read their settings from environment variables and have no defaults, so set them before running.

| Script | What it does | Required environment variables |
|---|---|---|
| `scripts/backup-runs.sh` | Uploads finished seeds from `runs/phase2` to a Cloud Storage bucket, shows backup status, or restores them. | `GCP_PROJECT`, `GCS_BUCKET`, plus `GCS_LOCATION` only if the bucket must be created |
| `scripts/clean_training.sh` | Stops leftover training processes and moves incomplete seed folders to the trash. | None |
| `scripts/gcp-deploy.sh` | Creates, monitors and destroys a Compute Engine virtual machine that trains the ablation. | `GCP_PROJECT`, `GCP_ZONE`, `GCP_VM_NAME` (and `GCS_BUCKET` for `download`) |
| `scripts/safe-delete.sh` | Moves files to the macOS Trash instead of deleting them, with a restore option. | None |
| `.python-version` | Pins Python 3.12 for `uv`. | None |

### Compare two systems (Phase 3)

Once analysis results exist for both an RNN run and a biological dataset, compare their signatures:

```bash
uv run geometric-signatures compare runs/rnn/ runs/bio/ --output figures/
```

Loaders for the public datasets live in `src/geometric_signatures/data/` (`ibl.py`, `allen.py`).

## Results summary (Phase 2)

<!-- TODO: drop actual numbers once you write up the chapter. -->

| Motif | RSA-based separability | Persistent-homology separability | MARBLE separability |
|---|:---:|:---:|:---:|
| Normalization / gain modulation | ✅ | ✅ | ✅ |
| Attractor dynamics | ✅ | ✅ | ⚠️ partial |
| Selective gating | ✅ | ⚠️ partial | ✅ |
| Expansion recoding | ✅ | ⚠️ partial | ✅ |

Across 50 seeds per condition, between-motif silhouette score is `>0.6` for all motifs on the dominant tool, while within-motif scatter stays small. Full numbers in `figures/phase2/comparisons.json`.

## What's next

- **Phase 3 IBL** — apply the signature pipeline to IBL sessions where the task structure plausibly induces each motif. Target: 4 motifs × 30 sessions.
- **Phase 3 Allen Brain Observatory** — same pipeline on visual cortex under structured stimuli. Tests whether the motifs generalize beyond decision-making cortex.
- **Theoretical chapter** — formal proof of which population geometries are *necessary* vs. *sufficient* for each motif.
- **Practical tool** — package the diagnostic as a one-shot analysis users can apply to their own population recordings.

## Citation

If this codebase or the underlying work is useful, please cite:

```bibtex
@phdthesis{lopez_geometric_signatures_2026,
  author  = {Lopez, Jose B.},
  title   = {Geometric Signatures of Computational Motifs in Neural Population Dynamics},
  school  = {[TODO: institution]},
  year    = {2026},
  note    = {In progress; codebase: https://github.com/aifriend/geometric-signatures-proposal}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Jose Lopez** — AI engineer in Madrid, working on the intersection of biological and artificial intelligence.

- GitHub: [@aifriend](https://github.com/aifriend)
- LinkedIn: [jafdl](https://www.linkedin.com/in/jafdl)
- Website: [auto-latam.com](https://auto-latam.com/en)

## Acknowledgments

- The MARBLE library authors for the geometric-invariant tooling
- The International Brain Laboratory and Allen Institute for releasing the population recordings that make Phase 3 even possible
- The wider population-dynamics community (Vyas, Golub, Sussillo, Maheswaranathan, and others) whose work motivated the choice of motifs
