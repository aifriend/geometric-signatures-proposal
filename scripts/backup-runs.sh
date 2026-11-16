#!/usr/bin/env bash
# backup-runs.sh — Backup training runs to Google Cloud Storage.
#
# Uploads completed seeds (manifest.json + population.npz) to a GCS bucket.
# Designed to be run periodically during long training sessions.
#
# Usage:
#   ./scripts/backup-runs.sh                    # upload new/changed files
#   ./scripts/backup-runs.sh --status           # show what's backed up vs local
#   ./scripts/backup-runs.sh --restore          # download all from GCS → local
#   ./scripts/backup-runs.sh --restore-seed <variant> <seed>  # restore one seed
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCS bucket (auto-created if missing)
#
# Required environment variables (no defaults):
#   GCP_PROJECT   Google Cloud project ID that owns the bucket
#   GCS_BUCKET    Bucket name, without the gs:// prefix
# Needed only when the bucket has to be created:
#   GCS_LOCATION  Bucket location, e.g. a region name

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="${PROJECT_DIR}/runs/phase2"
: "${GCP_PROJECT:?set GCP_PROJECT (Google Cloud project ID)}"
: "${GCS_BUCKET:?set GCS_BUCKET (bucket name, without gs://)}"
GCS_PROJECT="$GCP_PROJECT"
GCS_BUCKET="gs://${GCS_BUCKET#gs://}"
GCS_PREFIX="${GCS_BUCKET}/runs/phase2"

# What files to backup per seed (essential data only, skip large checkpoints)
BACKUP_FILES=("manifest.json" "population.npz")

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${BLUE}ℹ${NC}  $*"; }
log_ok()    { echo -e "${GREEN}✓${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}⚠${NC}  $*"; }
log_error() { echo -e "${RED}✗${NC}  $*" >&2; }

# ──────────────────────────────────────────────────────────────────────
# Ensure bucket exists
# ──────────────────────────────────────────────────────────────────────
ensure_bucket() {
    if ! gsutil ls "$GCS_BUCKET" &>/dev/null; then
        log_info "Creating GCS bucket: ${GCS_BUCKET}"
        gsutil mb -p "$GCS_PROJECT" -l "${GCS_LOCATION:?set GCS_LOCATION to create the bucket}" -b on "$GCS_BUCKET"
        log_ok "Bucket created"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Check gcloud auth
# ──────────────────────────────────────────────────────────────────────
check_auth() {
    if ! gcloud auth print-access-token &>/dev/null; then
        log_error "Not authenticated. Run: gcloud auth login"
        exit 1
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Upload: sync completed seeds to GCS
# ──────────────────────────────────────────────────────────────────────
cmd_upload() {
    echo -e "${BOLD}╔═════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║   Backup Training Runs → GCS                       ║${NC}"
    echo -e "${BOLD}╚═════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [ ! -d "$RUNS_DIR" ]; then
        log_warn "No runs directory found: ${RUNS_DIR}"
        echo "  Nothing to backup."
        return 0
    fi

    check_auth
    ensure_bucket

    local uploaded=0
    local skipped=0
    local failed=0

    # Find all completed seeds
    for seed_dir in $(find "$RUNS_DIR" -type d -name "seed_*" 2>/dev/null | sort); do
        local manifest="${seed_dir}/manifest.json"
        local population="${seed_dir}/population.npz"

        # Only backup complete seeds
        if [ ! -f "$manifest" ] || [ ! -f "$population" ]; then
            continue
        fi

        # Relative path from RUNS_DIR
        local rel_path="${seed_dir#${RUNS_DIR}/}"

        # Check if already backed up (compare sizes)
        local remote_manifest="${GCS_PREFIX}/${rel_path}/manifest.json"
        local local_size
        local_size=$(wc -c < "$manifest" | tr -d ' ')
        local remote_size
        remote_size=$(gsutil ls -l "$remote_manifest" 2>/dev/null | head -1 | awk '{print $1}' || echo "0")

        if [ "$remote_size" = "$local_size" ] && [ "$remote_size" != "0" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        # Upload essential files
        log_info "Uploading: ${rel_path}"
        for fname in "${BACKUP_FILES[@]}"; do
            local local_file="${seed_dir}/${fname}"
            if [ -f "$local_file" ]; then
                if gsutil -q cp "$local_file" "${GCS_PREFIX}/${rel_path}/${fname}" 2>/dev/null; then
                    true
                else
                    log_error "Failed to upload: ${rel_path}/${fname}"
                    failed=$((failed + 1))
                fi
            fi
        done

        # Also backup analysis results if they exist
        local analysis_dir="${seed_dir}/analysis"
        if [ -d "$analysis_dir" ]; then
            gsutil -q -m cp -r "$analysis_dir" "${GCS_PREFIX}/${rel_path}/" 2>/dev/null || true
        fi

        uploaded=$((uploaded + 1))
    done

    echo ""
    echo -e "${BOLD}── Summary ──${NC}"
    echo -e "  Uploaded:  ${GREEN}${uploaded}${NC} seeds"
    echo -e "  Skipped:   ${BLUE}${skipped}${NC} (already backed up)"
    if [ "$failed" -gt 0 ]; then
        echo -e "  Failed:    ${RED}${failed}${NC}"
    fi
    echo -e "  Bucket:    ${BLUE}${GCS_PREFIX}${NC}"

    # Also upload the config and comparison results
    local config="${PROJECT_DIR}/config/experiment.ablation_template.yaml"
    if [ -f "$config" ]; then
        gsutil -q cp "$config" "${GCS_BUCKET}/config/" 2>/dev/null || true
    fi
    local comparisons="${RUNS_DIR}/comparisons.json"
    if [ -f "$comparisons" ]; then
        gsutil -q cp "$comparisons" "${GCS_PREFIX}/comparisons.json" 2>/dev/null || true
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Status: compare local vs GCS
# ──────────────────────────────────────────────────────────────────────
cmd_status() {
    echo -e "${BOLD}Backup Status — Local vs GCS${NC}"
    echo ""

    check_auth

    # Count local completed
    local local_count=0
    if [ -d "$RUNS_DIR" ]; then
        local_count=$(find "$RUNS_DIR" -name manifest.json 2>/dev/null | wc -l | tr -d ' ')
    fi

    # Count remote completed
    local remote_count=0
    remote_count=$(gsutil ls "${GCS_PREFIX}/**/manifest.json" 2>/dev/null | wc -l | tr -d ' ')

    echo -e "  Local completed seeds:   ${BLUE}${local_count}${NC}/50"
    echo -e "  GCS backed up seeds:     ${GREEN}${remote_count}${NC}/50"

    if [ "$local_count" -gt "$remote_count" ]; then
        local diff=$((local_count - remote_count))
        echo -e "  ${YELLOW}⚠ ${diff} seeds NOT backed up. Run: $0${NC}"
    elif [ "$local_count" -eq "$remote_count" ] && [ "$local_count" -gt 0 ]; then
        echo -e "  ${GREEN}✓ All local seeds are backed up${NC}"
    fi

    # Show bucket size
    if [ "$remote_count" -gt 0 ]; then
        echo ""
        local total_size
        total_size=$(gsutil du -s "$GCS_PREFIX" 2>/dev/null | awk '{
            val=$1;
            if (val > 1073741824) printf "%.1f GB", val/1073741824;
            else if (val > 1048576) printf "%.1f MB", val/1048576;
            else printf "%.1f KB", val/1024;
        }')
        echo -e "  GCS total size: ${total_size}"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Restore: download all from GCS → local
# ──────────────────────────────────────────────────────────────────────
cmd_restore() {
    echo -e "${BOLD}Restoring Training Runs from GCS${NC}"
    echo ""

    check_auth

    # Check what's in GCS
    local remote_count
    remote_count=$(gsutil ls "${GCS_PREFIX}/**/manifest.json" 2>/dev/null | wc -l | tr -d ' ')

    if [ "$remote_count" -eq 0 ]; then
        log_warn "No backups found in GCS"
        log_info "Bucket: ${GCS_PREFIX}"
        return 1
    fi

    log_info "Found ${remote_count} backed up seeds in GCS"

    # Create local dirs
    mkdir -p "$RUNS_DIR"

    # Download
    log_info "Downloading to: ${RUNS_DIR}/"
    gsutil -m cp -r "${GCS_PREFIX}/*" "$RUNS_DIR/" 2>/dev/null

    # Verify
    local local_count
    local_count=$(find "$RUNS_DIR" -name manifest.json 2>/dev/null | wc -l | tr -d ' ')
    log_ok "Restored ${local_count} seeds to ${RUNS_DIR}/"
}

# ──────────────────────────────────────────────────────────────────────
# Restore single seed
# ──────────────────────────────────────────────────────────────────────
cmd_restore_seed() {
    local variant="$1"
    local seed="$2"

    check_auth

    local remote_path="${GCS_PREFIX}/${variant}/seed_${seed}"
    local local_path="${RUNS_DIR}/${variant}/seed_${seed}"

    log_info "Restoring: ${variant}/seed_${seed}"

    # Check remote exists
    if ! gsutil ls "${remote_path}/manifest.json" &>/dev/null; then
        log_error "Not found in backup: ${variant}/seed_${seed}"
        return 1
    fi

    mkdir -p "$local_path"
    gsutil -m cp -r "${remote_path}/*" "$local_path/"

    log_ok "Restored: ${local_path}/"
}

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
case "${1:---upload}" in
    --upload|-u|"")
        cmd_upload
        ;;
    --status|-s)
        cmd_status
        ;;
    --restore|-r)
        cmd_restore
        ;;
    --restore-seed)
        if [ $# -lt 3 ]; then
            log_error "Usage: $0 --restore-seed <variant> <seed>"
            exit 1
        fi
        cmd_restore_seed "$2" "$3"
        ;;
    --help|-h)
        echo "Usage: $0 [--upload|--status|--restore|--restore-seed <variant> <seed>]"
        echo ""
        echo "  (default)                  Upload completed seeds to GCS"
        echo "  --status                   Compare local vs GCS"
        echo "  --restore                  Download all from GCS → local"
        echo "  --restore-seed <v> <s>     Download one seed"
        echo ""
        echo "Bucket: ${GCS_BUCKET}"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use: $0 --help"
        exit 1
        ;;
esac
