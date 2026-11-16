#!/usr/bin/env bash
# gcp-deploy.sh — Deploy and manage Phase 2 training on GCP Compute Engine
#
# Full lifecycle: create VM → upload code → setup → train → monitor → download → destroy
#
# Usage:
#   ./scripts/gcp-deploy.sh create      # Create VM + upload code + install deps + launch training
#   ./scripts/gcp-deploy.sh upload      # Upload local code to VM (after local changes)
#   ./scripts/gcp-deploy.sh status      # Check training progress
#   ./scripts/gcp-deploy.sh logs        # Stream training logs (Ctrl+C to exit)
#   ./scripts/gcp-deploy.sh ssh         # Interactive SSH session
#   ./scripts/gcp-deploy.sh download    # Download completed runs to local + backup to GCS
#   ./scripts/gcp-deploy.sh restart     # Clean + relaunch training (after crash/preemption)
#   ./scripts/gcp-deploy.sh destroy     # Auto-download + delete VM
#   ./scripts/gcp-deploy.sh help        # Show help
#
# Required environment variables (no defaults):
#   GCP_PROJECT   Google Cloud project ID used for the VM
#   GCP_ZONE      Compute Engine zone for the VM, e.g. europe-west1-b
#   GCP_VM_NAME   Name of the training VM to create/manage
# The 'download' command also runs backup-runs.sh, which needs GCS_BUCKET.
#
# Cost: ~$0.73 total (n2-standard-8 Spot, ~$0.06/hr, ~10-12h)
# Note: CPU-only — model too small (256 hidden units) to benefit from GPU

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Config — edit these if needed
# ──────────────────────────────────────────────────────────────────────
: "${GCP_PROJECT:?set GCP_PROJECT (Google Cloud project ID)}"
: "${GCP_ZONE:?set GCP_ZONE (Compute Engine zone)}"
: "${GCP_VM_NAME:?set GCP_VM_NAME (training VM name)}"
PROJECT_ID="$GCP_PROJECT"
ZONE="$GCP_ZONE"
VM_NAME="$GCP_VM_NAME"
MACHINE_TYPE="n2-standard-8"                    # 8 vCPU, 32GB RAM (CPU-only)
IMAGE_FAMILY="ubuntu-2204-lts"                  # Standard Ubuntu (no GPU overhead)
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="30GB"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="geometric-signatures"                                # Directory on the VM
STATE_FILE="${PROJECT_DIR}/.gcp-vm-name"         # Local file tracking active VM

# Files to exclude from rsync upload
RSYNC_EXCLUDE=(
    "runs/"
    ".venv/"
    "__pycache__/"
    "*.pyc"
    ".git/"
    ".mypy_cache/"
    ".pytest_cache/"
    "plan.md"
    "*.egg-info/"
    "/tmp/"
)

# ──────────────────────────────────────────────────────────────────────
# Colors + logging
# ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${BLUE}ℹ${NC}  $*"; }
log_ok()    { echo -e "${GREEN}✓${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}⚠${NC}  $*"; }
log_error() { echo -e "${RED}✗${NC}  $*" >&2; }

# ──────────────────────────────────────────────────────────────────────
# State management — track which VM is active
# ──────────────────────────────────────────────────────────────────────
save_state() { echo "$VM_NAME" > "$STATE_FILE"; }
load_state() { [ -f "$STATE_FILE" ] && VM_NAME=$(cat "$STATE_FILE") || true; }
clear_state() { rm -f "$STATE_FILE"; }

require_vm() {
    load_state
    if ! gcloud compute instances describe "$VM_NAME" \
            --project="$PROJECT_ID" --zone="$ZONE" &>/dev/null; then
        log_error "VM '$VM_NAME' not found. Run: ./scripts/gcp-deploy.sh create"
        return 1
    fi
}

# Helper: run a command on the remote VM via SSH (IAP tunnel for reliability)
remote() {
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --tunnel-through-iap \
        --command="$1"
}

# Helper: scp via IAP tunnel
remote_scp_up() {
    gcloud compute scp \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --tunnel-through-iap \
        "$1" \
        "$VM_NAME:$2"
}

remote_scp_down() {
    gcloud compute scp \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --tunnel-through-iap \
        "$VM_NAME:$1" \
        "$2"
}

# ──────────────────────────────────────────────────────────────────────
# upload — Sync local code to VM via gcloud scp
# ──────────────────────────────────────────────────────────────────────
cmd_upload() {
    require_vm

    log_info "Uploading local code to VM..."

    # Build exclude args
    local exclude_args=""
    for pattern in "${RSYNC_EXCLUDE[@]}"; do
        exclude_args="${exclude_args} --exclude=${pattern}"
    done

    # Create tar of local code (excluding heavy dirs)
    local tmp_tar="/tmp/geosig-code-upload.tar.gz"
    cd "$PROJECT_DIR"
    tar czf "$tmp_tar" \
        --exclude='runs' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='.mypy_cache' \
        --exclude='.pytest_cache' \
        --exclude='plan.md' \
        --exclude='*.egg-info' \
        --exclude='uv.lock' \
        .

    # Upload and extract
    remote_scp_up "$tmp_tar" "/tmp/geosig-code-upload.tar.gz"

    remote '
mkdir -p '"$REMOTE_DIR"'
cd '"$REMOTE_DIR"'
tar xzf /tmp/geosig-code-upload.tar.gz
rm -f /tmp/geosig-code-upload.tar.gz
echo "Code uploaded. Files:"
ls -la
'

    rm -f "$tmp_tar"
    log_ok "Code uploaded to VM:~/${REMOTE_DIR}/"
}

# ──────────────────────────────────────────────────────────────────────
# create — Provision VM + upload code + setup env + start training
# ──────────────────────────────────────────────────────────────────────
cmd_create() {
    echo -e "${BOLD}╔═════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║   GCP Phase 2 Training — Create & Launch (CPU)     ║${NC}"
    echo -e "${BOLD}╚═════════════════════════════════════════════════════╝${NC}"
    echo ""

    load_state
    if gcloud compute instances describe "$VM_NAME" \
            --project="$PROJECT_ID" --zone="$ZONE" &>/dev/null; then
        log_warn "VM already exists: $VM_NAME"
        log_info "Use: ./scripts/gcp-deploy.sh status"
        log_info "Or:  ./scripts/gcp-deploy.sh destroy  (to start fresh)"
        return 0
    fi

    log_info "Creating spot instance: $VM_NAME"
    log_info "  Machine:  $MACHINE_TYPE (8 vCPU, 32GB RAM, CPU-only)"
    log_info "  Zone:     $ZONE"
    log_info "  Image:    $IMAGE_FAMILY (Ubuntu 22.04 LTS)"
    log_info "  Pricing:  spot (~\$0.06/hr)"
    log_info "  Est. cost: ~\$0.73 total (~10-12h)"
    echo ""

    # Step 1: Create VM (no GPU)
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --maintenance-policy=TERMINATE \
        --scopes=cloud-platform \
        --quiet

    save_state
    log_ok "VM created"

    # Step 2: Wait for SSH to be available
    log_info "Waiting for SSH..."
    for i in $(seq 1 30); do
        if gcloud compute ssh "$VM_NAME" \
                --project="$PROJECT_ID" --zone="$ZONE" \
                --command="echo ok" &>/dev/null; then
            break
        fi
        sleep 5
    done

    # Step 3: Upload local code
    echo ""
    cmd_upload

    # Step 4: Setup environment
    log_info "Setting up environment (install uv + deps, ~3 min)..."
    remote '
set -e

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd '"$REMOTE_DIR"'

# Install Python + deps (CPU-only PyTorch)
uv sync --extra dev

# Install analysis deps (needed for future pipeline)
uv pip install scipy scikit-learn ripser rsatoolbox

# Verify PyTorch
uv run python -c "import torch; print(f\"PyTorch {torch.__version__}, CPU threads: {torch.get_num_threads()}\")"

# Set PyTorch to use all 8 cores
echo "export OMP_NUM_THREADS=8" >> ~/.bashrc
echo "export MKL_NUM_THREADS=8" >> ~/.bashrc

# Create output dir
mkdir -p runs/phase2

echo ""
echo "=== Setup complete ==="
'
    log_ok "Environment ready"

    # Step 5: Launch training
    echo ""
    log_info "Launching training..."
    cmd_launch

    echo ""
    echo -e "${BOLD}All done! Monitor with:${NC}"
    echo -e "  ${BLUE}./scripts/gcp-deploy.sh logs${NC}      # stream logs"
    echo -e "  ${BLUE}./scripts/gcp-deploy.sh status${NC}    # progress report"
}

# ──────────────────────────────────────────────────────────────────────
# launch — Start training process on the VM
# ──────────────────────────────────────────────────────────────────────
cmd_launch() {
    require_vm

    log_info "Starting training on $VM_NAME (CPU, 8 threads)..."

    remote '
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd '"$REMOTE_DIR"'

# Ensure no stale processes
pkill -f "geometric-signatures" 2>/dev/null || true
sleep 1

mkdir -p runs/phase2

nohup uv run geometric-signatures train \
    config/experiment.ablation_template.yaml \
    --output-dir runs/phase2 \
    --device cpu \
    > runs/phase2/training.log 2>&1 &

sleep 3
if pgrep -f "geometric-signatures" >/dev/null; then
    echo "Training started (PID: $(pgrep -f geometric-signatures | head -1))"
    tail -5 runs/phase2/training.log 2>/dev/null || true
else
    echo "ERROR: Training failed to start. Check logs:"
    cat runs/phase2/training.log 2>/dev/null || true
    exit 1
fi
'
    log_ok "Training launched"
}

# ──────────────────────────────────────────────────────────────────────
# status — Check training progress
# ──────────────────────────────────────────────────────────────────────
cmd_status() {
    require_vm

    echo -e "${BOLD}Training Status — $VM_NAME${NC}"
    echo ""

    remote '
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd '"$REMOTE_DIR"'

# Use clean_training.sh if available, fallback to manual check
if [ -x scripts/clean_training.sh ]; then
    ./scripts/clean_training.sh --status
else
    echo "Completed seeds:"
    find runs/phase2 -name manifest.json 2>/dev/null | wc -l
    echo ""
    echo "Active processes:"
    pgrep -af "geometric-signatures" || echo "  (none)"
    echo ""
    echo "Last log lines:"
    tail -3 runs/phase2/training.log 2>/dev/null || echo "  (no log)"
fi
'
}

# ──────────────────────────────────────────────────────────────────────
# logs — Stream training log in real-time
# ──────────────────────────────────────────────────────────────────────
cmd_logs() {
    require_vm
    log_info "Streaming logs from $VM_NAME (Ctrl+C to exit)..."
    echo ""
    remote 'tail -f '"$REMOTE_DIR"'/runs/phase2/training.log'
}

# ──────────────────────────────────────────────────────────────────────
# ssh — Interactive SSH session
# ──────────────────────────────────────────────────────────────────────
cmd_ssh() {
    require_vm
    log_info "Connecting to $VM_NAME..."
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --tunnel-through-iap
}

# ──────────────────────────────────────────────────────────────────────
# download — Download completed runs to local + backup to GCS
# ──────────────────────────────────────────────────────────────────────
cmd_download() {
    require_vm

    local dest="${PROJECT_DIR}/runs/phase2"
    mkdir -p "$dest"

    log_info "Downloading results from $VM_NAME → $dest/"

    # Count remote files
    remote '
cd '"$REMOTE_DIR"'
completed=$(find runs/phase2 -name manifest.json 2>/dev/null | wc -l)
echo "Completed seeds on VM: ${completed}/50"
'

    # Download only essential files (manifest.json + population.npz)
    # Using tar to transfer only needed files
    remote '
cd '"$REMOTE_DIR"'
find runs/phase2 \( -name manifest.json -o -name population.npz \) -print0 | \
    tar czf /tmp/training-results.tar.gz --null -T -
echo "Archive created: $(du -h /tmp/training-results.tar.gz | cut -f1)"
'

    remote_scp_down "/tmp/training-results.tar.gz" "/tmp/training-results.tar.gz"

    cd "$PROJECT_DIR"
    tar xzf /tmp/training-results.tar.gz
    rm -f /tmp/training-results.tar.gz

    local n_downloaded
    n_downloaded=$(find "$dest" -name manifest.json 2>/dev/null | wc -l | tr -d ' ')
    log_ok "Downloaded ${n_downloaded} seeds to: $dest/"

    # Auto-backup to GCS after download
    echo ""
    log_info "Backing up to GCS..."
    if [ -x "${PROJECT_DIR}/scripts/backup-runs.sh" ]; then
        "${PROJECT_DIR}/scripts/backup-runs.sh"
    else
        log_warn "backup-runs.sh not found, skipping GCS backup"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# restart — Clean + relaunch (after crash or spot preemption)
# ──────────────────────────────────────────────────────────────────────
cmd_restart() {
    require_vm

    # Check if VM is stopped (spot preemption)
    local vm_status
    vm_status=$(gcloud compute instances describe "$VM_NAME" \
        --project="$PROJECT_ID" --zone="$ZONE" \
        --format="value(status)" 2>/dev/null)

    if [ "$vm_status" = "TERMINATED" ] || [ "$vm_status" = "STOPPED" ]; then
        log_warn "VM was stopped (likely spot preemption). Restarting..."
        gcloud compute instances start "$VM_NAME" \
            --project="$PROJECT_ID" --zone="$ZONE" --quiet
        log_info "Waiting for VM to boot..."
        sleep 30
    fi

    # Upload latest code
    cmd_upload

    log_info "Cleaning + relaunching training on $VM_NAME..."
    echo ""

    remote '
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd '"$REMOTE_DIR"'

# Kill any stale processes
pkill -f "geometric-signatures" 2>/dev/null || true
sleep 2

# Remove incomplete seeds only
for d in $(find runs/phase2 -type d -name "seed_*" 2>/dev/null); do
    if [ ! -f "$d/manifest.json" ] || [ ! -f "$d/population.npz" ]; then
        rm -rf "$d"
        echo "Removed incomplete: $d"
    fi
done

echo ""
echo "Completed seeds preserved:"
find runs/phase2 -name manifest.json 2>/dev/null | wc -l

echo ""
echo "Relaunching training..."

mkdir -p runs/phase2
nohup uv run geometric-signatures train \
    config/experiment.ablation_template.yaml \
    --output-dir runs/phase2 \
    --device cpu \
    > runs/phase2/training.log 2>&1 &

sleep 3
echo "Done. PID: $(pgrep -f geometric-signatures | head -1)"
'

    log_ok "Training restarted"
}

# ──────────────────────────────────────────────────────────────────────
# destroy — Download results + backup to GCS + delete VM
# ──────────────────────────────────────────────────────────────────────
cmd_destroy() {
    require_vm

    echo -e "${RED}${BOLD}This will DELETE the VM: $VM_NAME${NC}"
    echo ""

    # Show what's on the VM
    log_info "Checking remote state before deletion..."
    remote '
cd '"$REMOTE_DIR"' 2>/dev/null || exit 0
completed=$(find runs/phase2 -name manifest.json 2>/dev/null | wc -l)
echo "  Completed runs on VM: $completed/50"
' || true

    echo ""
    read -p "Download results, backup to GCS, and delete VM? [yes/no]: " confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Cancelled"
        return 0
    fi

    # Download before destroying
    log_info "Downloading results + GCS backup..."
    cmd_download || log_warn "Download failed — VM may already be stopped"

    # Delete VM
    log_info "Deleting VM..."
    gcloud compute instances delete "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --quiet

    clear_state
    log_ok "VM deleted. Results in: runs/phase2/ + GCS backup"
}

# ──────────────────────────────────────────────────────────────────────
# help
# ──────────────────────────────────────────────────────────────────────
cmd_help() {
    cat << 'HELP'
gcp-deploy.sh — Manage Phase 2 training on GCP Compute Engine (CPU-only)

COMMANDS
  create      Create VM + upload code + setup + launch training
  upload      Upload local code to VM (for code changes)
  status      Check training progress
  logs        Stream training logs live (Ctrl+C to exit)
  ssh         Interactive SSH session on the VM
  download    Download completed runs to local + auto-backup to GCS
  restart     Handle crash/preemption: restart VM + upload code + clean + relaunch
  destroy     Download results + GCS backup + delete VM
  help        Show this help

TYPICAL WORKFLOW
  ./scripts/gcp-deploy.sh create       # ~5 min: provision + upload + setup + launch
  ./scripts/gcp-deploy.sh status       # check progress anytime
  ./scripts/gcp-deploy.sh logs         # stream log (Ctrl+C to exit)
  ./scripts/gcp-deploy.sh restart      # if preempted or crashed
  ./scripts/gcp-deploy.sh destroy      # auto-downloads + GCS backup + deletes VM

COST
  n2-standard-8 Spot: ~$0.06/hr (CPU-only, no GPU needed)
  Full training (50 seeds): ~$0.73 USD (~10-12 hours)

SPOT PREEMPTION
  If GCP reclaims the VM (~5% chance):
  1. VM stops automatically (instance-termination-action=STOP)
  2. Run: ./scripts/gcp-deploy.sh restart
  3. Training resumes from last completed seed

CODE SYNC
  Uses local code upload (tar + scp), NOT git clone.
  This means uncommitted local changes ARE included.
  Run 'upload' after local code changes to sync to VM.

FILES
  .gcp-vm-name    Local state file tracking active VM name
  runs/phase2/    Downloaded results (after download/destroy)

HELP
}

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    create)     cmd_create ;;
    upload)     cmd_upload ;;
    launch)     cmd_launch ;;
    status)     cmd_status ;;
    logs)       cmd_logs ;;
    ssh)        cmd_ssh ;;
    download)   cmd_download ;;
    restart)    cmd_restart ;;
    destroy)    cmd_destroy ;;
    help|-h|--help) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
