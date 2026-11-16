#!/usr/bin/env bash
# clean_training.sh — Prepare a safe re-run of Phase 2 training.
#
# Solves three recurring problems:
#   1. Zombie training processes from previous sessions competing for the same seeds
#   2. Incomplete/corrupted seed directories (checkpoint exists but no manifest)
#   3. Need a clear status report before relaunching
#
# Usage:
#   ./scripts/clean_training.sh                  # default: kill procs + clean incomplete
#   ./scripts/clean_training.sh --dry-run        # show what would be done, change nothing
#   ./scripts/clean_training.sh --full           # also remove ALL completed runs (fresh start)
#   ./scripts/clean_training.sh --status         # just report, touch nothing
#
# After running this, launch training with:
#   nohup uv run geometric-signatures train config/experiment.ablation_template.yaml \
#         --output-dir runs/phase2 --device mps > runs/phase2/training.log 2>&1 &

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="${PROJECT_DIR}/runs/phase2"
PROC_PATTERN="geometric-signatures"
LOG_FILE="${RUNS_DIR}/training.log"
SAFE_DELETE="${PROJECT_DIR}/scripts/safe-delete.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

# ──────────────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────────────
DRY_RUN=false
FULL_CLEAN=false
STATUS_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --full)     FULL_CLEAN=true ;;
        --status)   STATUS_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--full] [--status]"
            echo ""
            echo "  (default)    Kill training processes + remove incomplete seeds"
            echo "  --dry-run    Show what would be done without changing anything"
            echo "  --full       Remove ALL runs (fresh start from scratch)"
            echo "  --status     Report only — touch nothing"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (use --help)" >&2
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
count_completed() {
    if [ -d "$RUNS_DIR" ]; then
        find "$RUNS_DIR" -name manifest.json 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

count_incomplete() {
    local count=0
    if [ -d "$RUNS_DIR" ]; then
        for seed_dir in $(find "$RUNS_DIR" -type d -name "seed_*" 2>/dev/null); do
            if [ ! -f "${seed_dir}/manifest.json" ] || [ ! -f "${seed_dir}/population.npz" ]; then
                count=$((count + 1))
            fi
        done
    fi
    echo "$count"
}

list_completed() {
    if [ -d "$RUNS_DIR" ]; then
        find "$RUNS_DIR" -name manifest.json 2>/dev/null | while read -r mf; do
            local seed_dir
            seed_dir="$(dirname "$mf")"
            local rel_path="${seed_dir#${RUNS_DIR}/}"
            # Extract val_acc and epochs from manifest
            local val_acc="?"
            local epochs="?"
            local variant="?"
            if command -v python3 &>/dev/null; then
                val_acc=$(python3 -c "
import json
m = json.load(open('${mf}'))
v = m.get('final_val_accuracy', m.get('best_val_acc', None))
print(f'{v:.1%}' if v is not None else '?')
" 2>/dev/null || echo "?")
                epochs=$(python3 -c "
import json
m = json.load(open('${mf}'))
print(m.get('epochs_completed', '?'))
" 2>/dev/null || echo "?")
                variant=$(python3 -c "
import json
m = json.load(open('${mf}'))
print(m.get('variant', '?'))
" 2>/dev/null || echo "?")
            fi
            echo "    ${rel_path}  (val_acc=${val_acc}, epochs=${epochs})"
        done | sort
    fi
}

list_incomplete() {
    if [ -d "$RUNS_DIR" ]; then
        for seed_dir in $(find "$RUNS_DIR" -type d -name "seed_*" 2>/dev/null | sort); do
            if [ ! -f "${seed_dir}/manifest.json" ] || [ ! -f "${seed_dir}/population.npz" ]; then
                local rel_path="${seed_dir#${RUNS_DIR}/}"
                local has_ckpt=""
                [ -f "${seed_dir}/checkpoint_best.pt" ] && has_ckpt=" [has checkpoint]"
                echo "    ${rel_path}${has_ckpt}"
            fi
        done
    fi
}

count_procs() {
    ps aux 2>/dev/null | grep "$PROC_PATTERN" | grep -v grep | grep python | wc -l | tr -d ' '
}

list_procs() {
    ps aux 2>/dev/null | grep "$PROC_PATTERN" | grep -v grep | grep python | \
        awk '{printf "    PID %-8s (started %s %s, cpu=%s%%)\n", $2, $9, $10, $3}'
}

# ──────────────────────────────────────────────────────────────────────
# Status report
# ──────────────────────────────────────────────────────────────────────
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Phase 2 Training — Cleanup & Status Report    ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Processes
n_procs=$(count_procs)
if [ "$n_procs" -gt 0 ]; then
    echo -e "${YELLOW}⚡ Active training processes: ${n_procs}${NC}"
    list_procs
else
    echo -e "${GREEN}✓ No active training processes${NC}"
fi
echo ""

# Completed runs
n_completed=$(count_completed)
echo -e "${BLUE}📦 Completed runs: ${n_completed}/50${NC}"
if [ "$n_completed" -gt 0 ]; then
    list_completed
fi
echo ""

# Incomplete runs
n_incomplete=$(count_incomplete)
if [ "$n_incomplete" -gt 0 ]; then
    echo -e "${YELLOW}⚠  Incomplete seeds: ${n_incomplete}${NC}"
    list_incomplete
else
    echo -e "${GREEN}✓ No incomplete seeds${NC}"
fi
echo ""

# Log file
if [ -f "$LOG_FILE" ]; then
    local_size=$(du -h "$LOG_FILE" 2>/dev/null | awk '{print $1}')
    echo -e "${BLUE}📄 Log file: ${LOG_FILE} (${local_size})${NC}"
    echo "    Last line: $(tail -1 "$LOG_FILE" 2>/dev/null | head -c 120)"
else
    echo -e "📄 No log file found"
fi
echo ""

# ETA
remaining=$((50 - n_completed))
if [ "$remaining" -gt 0 ]; then
    hours=$(echo "scale=1; $remaining * 57 / 60" | bc 2>/dev/null || echo "?")
    echo -e "${BLUE}⏱  Estimated time remaining: ~${hours} hours (${remaining} seeds × ~57 min)${NC}"
    echo ""
fi

# ──────────────────────────────────────────────────────────────────────
# Status-only mode: stop here
# ──────────────────────────────────────────────────────────────────────
if $STATUS_ONLY; then
    echo -e "${GREEN}(status-only mode — nothing modified)${NC}"
    exit 0
fi

# ──────────────────────────────────────────────────────────────────────
# Step 1: Kill training processes
# ──────────────────────────────────────────────────────────────────────
if [ "$n_procs" -gt 0 ]; then
    echo -e "${BOLD}── Step 1: Kill training processes ──${NC}"
    if $DRY_RUN; then
        echo -e "${YELLOW}  [DRY RUN] Would kill ${n_procs} process(es)${NC}"
    else
        # Kill Python workers first (SIGTERM), then parent uv processes
        ps aux | grep "$PROC_PATTERN" | grep -v grep | awk '{print $2}' | \
            xargs kill 2>/dev/null || true
        sleep 2

        # Verify
        n_remaining=$(count_procs)
        if [ "$n_remaining" -gt 0 ]; then
            echo -e "${YELLOW}  ${n_remaining} process(es) still alive, sending SIGKILL...${NC}"
            ps aux | grep "$PROC_PATTERN" | grep -v grep | awk '{print $2}' | \
                xargs kill -9 2>/dev/null || true
            sleep 1
        fi

        n_final=$(count_procs)
        if [ "$n_final" -eq 0 ]; then
            echo -e "${GREEN}  ✓ All training processes killed${NC}"
        else
            echo -e "${RED}  ✗ ${n_final} process(es) could not be killed${NC}"
        fi
    fi
    echo ""
fi

# ──────────────────────────────────────────────────────────────────────
# Step 2: Clean incomplete seeds
# ──────────────────────────────────────────────────────────────────────
if $FULL_CLEAN; then
    echo -e "${BOLD}── Step 2: Full clean (remove ALL runs via safe-delete) ──${NC}"
    if [ -d "$RUNS_DIR" ]; then
        # Find all variant directories (but preserve the phase2 dir and log)
        for variant_dir in "$RUNS_DIR"/*/; do
            [ -d "$variant_dir" ] || continue
            variant_name="$(basename "$variant_dir")"
            [ "$variant_name" = "logs" ] && continue  # preserve logs dir
            if $DRY_RUN; then
                n_seeds=$(find "$variant_dir" -type d -name "seed_*" | wc -l | tr -d ' ')
                echo -e "${YELLOW}  [DRY RUN] Would safe-delete ${variant_name}/ (${n_seeds} seeds)${NC}"
            else
                "$SAFE_DELETE" "$variant_dir"
            fi
        done
        # Clean log file (logs are ok to rm, not training data)
        if [ -f "$LOG_FILE" ]; then
            if $DRY_RUN; then
                echo -e "${YELLOW}  [DRY RUN] Would remove training.log${NC}"
            else
                rm -f "$LOG_FILE"
                echo -e "${GREEN}  ✓ Removed training.log${NC}"
            fi
        fi
    else
        echo -e "${GREEN}  Nothing to clean (${RUNS_DIR} does not exist)${NC}"
    fi
elif [ "$n_incomplete" -gt 0 ]; then
    echo -e "${BOLD}── Step 2: Remove incomplete seeds (via safe-delete) ──${NC}"
    if [ -d "$RUNS_DIR" ]; then
        cleaned=0
        for seed_dir in $(find "$RUNS_DIR" -type d -name "seed_*" 2>/dev/null | sort); do
            if [ ! -f "${seed_dir}/manifest.json" ] || [ ! -f "${seed_dir}/population.npz" ]; then
                rel_path="${seed_dir#${RUNS_DIR}/}"
                if $DRY_RUN; then
                    echo -e "${YELLOW}  [DRY RUN] Would safe-delete ${rel_path}${NC}"
                else
                    "$SAFE_DELETE" "$seed_dir"
                fi
                cleaned=$((cleaned + 1))
            fi
        done
        if [ "$cleaned" -eq 0 ]; then
            echo -e "${GREEN}  Nothing to clean${NC}"
        fi
    fi
else
    echo -e "${BOLD}── Step 2: Clean incomplete seeds ──${NC}"
    echo -e "${GREEN}  Nothing to clean${NC}"
fi
echo ""

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
echo -e "${BOLD}── Summary ──${NC}"
n_completed_after=$(count_completed)
n_remaining_after=$((50 - n_completed_after))

if $DRY_RUN; then
    echo -e "${YELLOW}  [DRY RUN] No changes were made${NC}"
fi

echo -e "  Completed runs preserved: ${GREEN}${n_completed_after}${NC}/50"
echo -e "  Remaining to train:       ${BLUE}${n_remaining_after}${NC}"
echo ""

if [ "$n_remaining_after" -gt 0 ] && ! $DRY_RUN; then
    echo -e "${BOLD}Ready to launch:${NC}"
    echo -e "  ${BLUE}nohup uv run geometric-signatures train \\\\${NC}"
    echo -e "  ${BLUE}  config/experiment.ablation_template.yaml \\\\${NC}"
    echo -e "  ${BLUE}  --output-dir runs/phase2 --device mps \\\\${NC}"
    echo -e "  ${BLUE}  > runs/phase2/training.log 2>&1 &${NC}"
    echo ""
    echo -e "Monitor with: ${BLUE}tail -f runs/phase2/training.log${NC}"
fi
