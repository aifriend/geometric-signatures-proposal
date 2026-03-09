"""Phase 2 Live Training Dashboard for Geometric Signatures.

Trains all 5 ablation variants x N seeds with live progress tracking.

Launch with:
    uv run streamlit run scripts/streamlit_dashboard.py
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import streamlit as st

# Short variant labels for the status grid
_SHORT_NAMES: dict[str, str] = {
    "complete": "Complete",
    "ablate_normalization_gain_modulation": "- Norm",
    "ablate_attractor_dynamics": "- Attractor",
    "ablate_selective_gating": "- Gating",
    "ablate_expansion_recoding": "- Expansion",
}

# ---------------------------------------------------------------------------
# Session state initialization (survives Streamlit re-runs)
# ---------------------------------------------------------------------------

if "run_state" not in st.session_state:
    st.session_state.run_state = "idle"  # idle | running | done | error | analyzing
    st.session_state.metrics_queue: queue.Queue = queue.Queue()
    st.session_state.run_progress_queue: queue.Queue = queue.Queue()
    st.session_state.cancel_event = threading.Event()

    # Per-run tracking:  variant -> seed -> [epoch_metrics_dicts]
    st.session_state.metrics_history: dict[str, dict[int, list[dict]]] = {}
    # variant -> seed -> "pending" | "running" | "done" | "error"
    st.session_state.run_status: dict[str, dict[int, str]] = {}

    st.session_state.current_variant: str | None = None
    st.session_state.current_seed: int | None = None
    st.session_state.total_epochs = 0
    st.session_state.total_runs = 0
    st.session_state.completed_runs = 0
    st.session_state.variant_names: list[str] = []
    st.session_state.seed_list: list[int] = []

    st.session_state.error_msg: str | None = None
    st.session_state.pipeline_result = None
    st.session_state.worker_thread: threading.Thread | None = None
    st.session_state.start_time: float = 0.0


# ---------------------------------------------------------------------------
# Worker thread — runs all variants x seeds then analysis
# ---------------------------------------------------------------------------


def _run_phase2_worker(
    config_path: str,
    output_dir: str,
    device: str,
    metrics_queue: queue.Queue,
    run_progress_queue: queue.Queue,
    cancel_event: threading.Event,
) -> None:
    """Train all ablation variants x seeds, then run analysis."""
    try:
        from geometric_signatures import load_experiment_config
        from geometric_signatures.motifs import build_single_ablation_variants
        from geometric_signatures.training import train_single_seed

        config = load_experiment_config(Path(config_path))
        variants = build_single_ablation_variants(config.motifs)

        if config.seeds is not None:
            seeds = list(config.seeds.seed_list())
        else:
            seeds = [config.experiment.seed]

        for variant_name, motifs in variants.items():
            if cancel_event.is_set():
                break

            variant_config = replace(config, motifs=motifs)

            for seed in seeds:
                if cancel_event.is_set():
                    break

                run_progress_queue.put((variant_name, seed, "start"))

                # Closure captures current variant_name/seed
                _vn, _sd = variant_name, seed

                def epoch_callback(m, vn=_vn, sd=_sd):  # type: ignore[assignment]
                    metrics_queue.put((vn, sd, m))

                try:
                    train_single_seed(
                        variant_config,
                        seed=seed,
                        output_dir=Path(output_dir),
                        device=device,
                        progress_callback=epoch_callback,
                        cancel_event=cancel_event,
                    )
                    run_progress_queue.put((variant_name, seed, "done"))
                except Exception as exc:
                    run_progress_queue.put(
                        (variant_name, seed, f"error:{exc}")
                    )

        # After all training, run analysis + aggregation + comparison
        if not cancel_event.is_set():
            run_progress_queue.put(("__pipeline__", 0, "analyzing"))
            try:
                from geometric_signatures.pipeline import (
                    PipelineOptions,
                    run_pipeline,
                )

                result = run_pipeline(
                    config,
                    Path(output_dir),
                    PipelineOptions(skip_training=True, device=device),
                )
                st.session_state.pipeline_result = result
            except Exception as exc:
                st.session_state.error_msg = f"Analysis failed: {exc}"

        st.session_state.run_state = "done"
    except Exception as exc:
        st.session_state.error_msg = str(exc)
        st.session_state.run_state = "error"


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Geometric Signatures — Phase 2 Dashboard", layout="wide"
)

st.sidebar.title("Phase 2 Configuration")

config_path = st.sidebar.text_input(
    "Config YAML path",
    value="config/experiment.ablation_template.yaml",
)
output_dir = st.sidebar.text_input(
    "Output directory",
    value="runs/phase2",
)
device = st.sidebar.selectbox("Device", ["auto", "cpu", "mps", "cuda"])

st.sidebar.markdown("---")

col_start, col_stop = st.sidebar.columns(2)

# Start button
start_disabled = st.session_state.run_state in ("running", "analyzing")
if col_start.button("Start Phase 2", disabled=start_disabled, use_container_width=True):
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        st.sidebar.error(f"Config not found: {cfg_path}")
    else:
        import yaml

        from geometric_signatures.motifs import build_single_ablation_variants

        raw = yaml.safe_load(cfg_path.read_text())
        total_epochs = raw.get("training", {}).get("epochs", 50)
        seeds_cfg = raw.get("seeds", {})
        n_seeds = seeds_cfg.get("n_seeds", 1)
        base_seed = seeds_cfg.get("base_seed", raw.get("experiment", {}).get("seed", 42))
        seed_list = list(range(base_seed, base_seed + n_seeds))

        # Build variant names from motif config
        from geometric_signatures.motifs import MotifSwitches

        motifs_raw = raw.get("motifs", {})
        base_motifs = MotifSwitches(**motifs_raw)
        variant_names = list(build_single_ablation_variants(base_motifs).keys())

        total_runs = len(variant_names) * len(seed_list)

        # Reset all state
        st.session_state.run_state = "running"
        st.session_state.error_msg = None
        st.session_state.pipeline_result = None
        st.session_state.total_epochs = total_epochs
        st.session_state.total_runs = total_runs
        st.session_state.completed_runs = 0
        st.session_state.variant_names = variant_names
        st.session_state.seed_list = seed_list
        st.session_state.current_variant = None
        st.session_state.current_seed = None
        st.session_state.start_time = time.time()
        st.session_state.cancel_event.clear()

        # Init tracking dicts
        st.session_state.metrics_history = {
            v: {s: [] for s in seed_list} for v in variant_names
        }
        st.session_state.run_status = {
            v: {s: "pending" for s in seed_list} for v in variant_names
        }

        # Drain leftover queue items
        for q in (st.session_state.metrics_queue, st.session_state.run_progress_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Launch worker
        t = threading.Thread(
            target=_run_phase2_worker,
            args=(
                config_path,
                output_dir,
                device,
                st.session_state.metrics_queue,
                st.session_state.run_progress_queue,
                st.session_state.cancel_event,
            ),
            daemon=True,
        )
        t.start()
        st.session_state.worker_thread = t
        st.rerun()

# Stop button
stop_disabled = st.session_state.run_state not in ("running", "analyzing")
if col_stop.button("Stop", disabled=stop_disabled, use_container_width=True):
    st.session_state.cancel_event.set()
    if st.session_state.worker_thread is not None:
        st.session_state.worker_thread.join(timeout=5)
    st.session_state.run_state = "done"
    st.rerun()


# ---------------------------------------------------------------------------
# Drain queues into state
# ---------------------------------------------------------------------------

# Drain run progress events
while not st.session_state.run_progress_queue.empty():
    try:
        variant, seed, status = st.session_state.run_progress_queue.get_nowait()
        if variant == "__pipeline__":
            st.session_state.run_state = "analyzing"
            continue
        if status == "start":
            st.session_state.run_status[variant][seed] = "running"
            st.session_state.current_variant = variant
            st.session_state.current_seed = seed
        elif status == "done":
            st.session_state.run_status[variant][seed] = "done"
            st.session_state.completed_runs += 1
        elif status.startswith("error:"):
            st.session_state.run_status[variant][seed] = "error"
            st.session_state.completed_runs += 1
    except queue.Empty:
        break

# Drain epoch metrics
while not st.session_state.metrics_queue.empty():
    try:
        variant, seed, epoch_metrics = st.session_state.metrics_queue.get_nowait()
        st.session_state.metrics_history[variant][seed].append(
            asdict(epoch_metrics)
        )
    except queue.Empty:
        break


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Geometric Signatures — Phase 2 Dashboard")

run_state = st.session_state.run_state

# Status banner
if run_state == "idle":
    st.info("Configure parameters in the sidebar and click **Start Phase 2**.")
elif run_state == "running":
    st.warning("Training in progress...")
elif run_state == "analyzing":
    st.warning("Training complete. Running analysis, aggregation, and comparison...")
elif run_state == "done":
    completed = st.session_state.completed_runs
    total = st.session_state.total_runs
    if st.session_state.error_msg:
        st.error(f"Completed with errors: {st.session_state.error_msg}")
    else:
        st.success(f"Phase 2 complete — {completed}/{total} runs finished.")
elif run_state == "error":
    st.error(f"Pipeline failed: {st.session_state.error_msg}")


if run_state != "idle":
    total_runs = st.session_state.total_runs
    completed_runs = st.session_state.completed_runs
    variant_names = st.session_state.variant_names
    seed_list = st.session_state.seed_list
    total_epochs = st.session_state.total_epochs

    # ── Global progress ──
    if total_runs > 0:
        st.progress(min(completed_runs / total_runs, 1.0))

    elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", f"{completed_runs} / {total_runs}")
    col2.metric("Current", _SHORT_NAMES.get(
        st.session_state.current_variant or "", st.session_state.current_variant or "—"
    ))
    col3.metric("Seed", str(st.session_state.current_seed) if st.session_state.current_seed is not None else "—")
    col4.metric("Elapsed", elapsed_str)

    # ── Status grid (variants x seeds) ──
    st.subheader("Run Status Grid")

    status_map = {"pending": ".", "running": "~", "done": "+", "error": "X"}
    grid_data: dict[str, list[str]] = {}
    for v in variant_names:
        label = _SHORT_NAMES.get(v, v)
        grid_data[label] = [
            status_map.get(st.session_state.run_status.get(v, {}).get(s, "pending"), "?")
            for s in seed_list
        ]
    grid_df = pd.DataFrame(grid_data, index=[f"s{s}" for s in seed_list]).T
    st.dataframe(grid_df, width="stretch")
    st.caption(". = pending  ~ = running  + = done  X = error")

    # ── Active run: epoch-level detail ──
    cur_v = st.session_state.current_variant
    cur_s = st.session_state.current_seed
    if cur_v and cur_s is not None:
        cur_history = st.session_state.metrics_history.get(cur_v, {}).get(cur_s, [])
        if cur_history:
            st.subheader(
                f"Active: {_SHORT_NAMES.get(cur_v, cur_v)} / seed {cur_s}"
            )

            cur_df = pd.DataFrame(cur_history)
            n_epochs = len(cur_df)

            # Epoch progress bar
            if total_epochs > 0:
                st.progress(min(n_epochs / total_epochs, 1.0))

            c1, c2, c3 = st.columns(3)
            c1.metric("Epoch", f"{n_epochs} / {total_epochs}")
            c2.metric("Train Loss", f"{cur_df['loss'].iloc[-1]:.4f}")
            c3.metric("Val Loss", f"{cur_df['val_loss'].iloc[-1]:.4f}")

            # Loss curve for active run
            if "epoch" in cur_df.columns:
                loss_df = cur_df[["epoch", "loss", "val_loss"]].set_index("epoch")
                st.line_chart(loss_df)

            # Per-task accuracy
            if "val_task_accuracies" in cur_df.columns:
                acc_records = []
                for _, row in cur_df.iterrows():
                    acc_dict = row["val_task_accuracies"]
                    if isinstance(acc_dict, dict):
                        acc_records.append({"epoch": row["epoch"], **acc_dict})
                if acc_records:
                    acc_df = pd.DataFrame(acc_records).set_index("epoch")
                    st.line_chart(acc_df)

    # ── Completed runs: browse by variant ──
    done_variants = [
        v for v in variant_names
        if any(
            st.session_state.run_status.get(v, {}).get(s) == "done"
            for s in seed_list
        )
    ]
    if done_variants:
        st.subheader("Completed Runs")
        selected_variant = st.selectbox(
            "Select variant",
            done_variants,
            format_func=lambda v: _SHORT_NAMES.get(v, v),
        )
        if selected_variant:
            # Collect final val_loss per seed
            seed_summaries = []
            for s in seed_list:
                hist = st.session_state.metrics_history.get(selected_variant, {}).get(s, [])
                if hist:
                    final = hist[-1]
                    seed_summaries.append({
                        "seed": s,
                        "epochs": len(hist),
                        "final_loss": final["loss"],
                        "final_val_loss": final["val_loss"],
                        "best_val_loss": min(h["val_loss"] for h in hist),
                    })
            if seed_summaries:
                summary_df = pd.DataFrame(seed_summaries)
                st.dataframe(
                    summary_df.style.format({
                        "final_loss": "{:.4f}",
                        "final_val_loss": "{:.4f}",
                        "best_val_loss": "{:.4f}",
                    }),
                    width="stretch",
                )

                # Overlay val_loss curves for all seeds of this variant
                st.markdown("**Val Loss across seeds**")
                overlay: dict[str, list[float]] = {}
                for s in seed_list:
                    hist = st.session_state.metrics_history.get(selected_variant, {}).get(s, [])
                    if hist:
                        overlay[f"seed_{s}"] = [h["val_loss"] for h in hist]
                if overlay:
                    max_len = max(len(v) for v in overlay.values())
                    overlay_df = pd.DataFrame(
                        {k: v + [None] * (max_len - len(v)) for k, v in overlay.items()}
                    )
                    st.line_chart(overlay_df)

    # ── Pipeline results (after analysis) ──
    if st.session_state.pipeline_result is not None:
        result = st.session_state.pipeline_result
        st.subheader("Pipeline Results")
        st.markdown(
            f"**{result.n_variants}** variants, **{result.n_seeds}** seeds per variant"
        )

        # Aggregated metrics
        if result.aggregated_results:
            st.markdown("#### Aggregated Metrics (mean across seeds)")
            for v_name, method_aggs in result.aggregated_results.items():
                with st.expander(_SHORT_NAMES.get(v_name, v_name)):
                    for method_name, agg in method_aggs.items():
                        st.markdown(f"**{method_name}**")
                        for metric_name, value in agg.means.items():
                            ci = agg.confidence_intervals.get(metric_name)
                            ci_str = f" (CI: [{ci[0]:.4f}, {ci[1]:.4f}])" if ci else ""
                            st.markdown(f"- {metric_name}: {value:.4f}{ci_str}")

        # Comparison table
        if result.comparisons:
            st.markdown("#### Ablation vs Complete (permutation tests)")
            comp_rows = []
            for v_name, tests in result.comparisons.items():
                for key, test_result in tests.items():
                    comp_rows.append({
                        "variant": _SHORT_NAMES.get(v_name, v_name),
                        "metric": key,
                        "p_value": test_result.p_value,
                        "statistic": test_result.observed_statistic,
                        "significant": "Yes" if test_result.p_value < 0.05 else "",
                    })
            if comp_rows:
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(
                    comp_df.style.format({
                        "p_value": "{:.4f}",
                        "statistic": "{:.4f}",
                    }),
                    width="stretch",
                )

elif run_state == "idle":
    st.markdown(
        """
        ### Phase 2: Full Ablation Experiment

        1. Set the **config path** to your ablation template YAML
        2. Choose an **output directory** for all checkpoints
        3. Pick a **device** (auto detects GPU/MPS)
        4. Click **Start Phase 2** to train all 5 variants x 10 seeds

        The dashboard tracks each run in a status grid and shows live
        loss curves for the active training run. After all training completes,
        analysis, aggregation, and statistical comparison run automatically.
        """
    )


# ---------------------------------------------------------------------------
# Auto-refresh while running
# ---------------------------------------------------------------------------

if run_state in ("running", "analyzing"):
    time.sleep(1)
    st.rerun()
