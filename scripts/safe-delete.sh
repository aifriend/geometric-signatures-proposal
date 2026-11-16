#!/usr/bin/env bash
# safe-delete.sh — Move files/dirs to macOS Trash instead of rm -rf.
#
# NEVER permanently deletes anything. Files go to ~/.Trash/ with a
# timestamp suffix to avoid name collisions.
#
# Usage:
#   ./scripts/safe-delete.sh <path> [<path2> ...]
#   ./scripts/safe-delete.sh --list              # list items in trash from this project
#   ./scripts/safe-delete.sh --restore <name>    # restore from trash to original location
#
# Examples:
#   ./scripts/safe-delete.sh runs/phase2/ablate_gating/seed_3
#   ./scripts/safe-delete.sh runs/phase2/ablate_*/seed_5  # glob works

set -euo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

TRASH_DIR="${HOME}/.Trash"
PROJECT_TAG="geosig"  # prefix for trash items from this project

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
log_info()  { echo -e "${BLUE}ℹ${NC}  $*"; }
log_ok()    { echo -e "${GREEN}✓${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}⚠${NC}  $*"; }
log_error() { echo -e "${RED}✗${NC}  $*" >&2; }

usage() {
    echo "Usage: $0 <path> [<path2> ...]"
    echo "       $0 --list"
    echo "       $0 --restore <trash-name>"
    echo ""
    echo "Moves files/directories to macOS Trash (~/.Trash/) instead of deleting."
    echo "Items are prefixed with '${PROJECT_TAG}' and timestamped."
}

# ──────────────────────────────────────────────────────────────────────
# List project items in trash
# ──────────────────────────────────────────────────────────────────────
cmd_list() {
    echo -e "${BOLD}Items in Trash from this project:${NC}"
    local count=0
    for item in "${TRASH_DIR}/${PROJECT_TAG}"--*; do
        [ -e "$item" ] || continue
        local name
        name="$(basename "$item")"
        local size
        size=$(du -sh "$item" 2>/dev/null | awk '{print $1}')
        echo "  ${name}  (${size})"
        count=$((count + 1))
    done
    if [ "$count" -eq 0 ]; then
        echo "  (none)"
    else
        echo ""
        echo "  Total: ${count} items"
        echo "  Restore with: $0 --restore <name>"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Restore from trash
# ──────────────────────────────────────────────────────────────────────
cmd_restore() {
    local trash_name="$1"
    local trash_path="${TRASH_DIR}/${trash_name}"

    if [ ! -e "$trash_path" ]; then
        log_error "Not found in trash: ${trash_name}"
        log_info "Use: $0 --list  to see available items"
        return 1
    fi

    # Extract original path from the metadata file
    local meta="${trash_path}.meta"
    if [ -f "$meta" ]; then
        local orig_path
        orig_path=$(grep '^original_path=' "$meta" | cut -d= -f2-)
        if [ -n "$orig_path" ]; then
            local dest_dir
            dest_dir="$(dirname "$orig_path")"
            mkdir -p "$dest_dir"

            if [ -e "$orig_path" ]; then
                log_warn "Original location already exists: ${orig_path}"
                log_info "Move manually from: ${trash_path}"
                return 1
            fi

            mv "$trash_path" "$orig_path"
            rm -f "$meta"
            log_ok "Restored: ${orig_path}"
            return 0
        fi
    fi

    log_warn "No metadata found. Item is at: ${trash_path}"
    log_info "Move manually to the desired location."
}

# ──────────────────────────────────────────────────────────────────────
# Safe delete: move to trash
# ──────────────────────────────────────────────────────────────────────
safe_delete() {
    local target="$1"

    # Resolve to absolute path
    local abs_path
    abs_path="$(cd "$(dirname "$target")" 2>/dev/null && pwd)/$(basename "$target")"

    if [ ! -e "$abs_path" ]; then
        log_warn "Does not exist: ${target}"
        return 0
    fi

    # Safety: refuse to delete critical directories
    local basename_target
    basename_target="$(basename "$abs_path")"
    case "$abs_path" in
        "$HOME"|"$HOME/"|/|"$(dirname "$HOME")"|"$(dirname "$HOME")/")
            log_error "REFUSED: Will not delete system directory: ${abs_path}"
            return 1
            ;;
    esac

    # Extra safety: never delete the entire runs/ or runs/phase2/ directory
    if [[ "$abs_path" =~ /runs/?$ ]] || [[ "$abs_path" =~ /runs/phase2/?$ ]]; then
        log_error "REFUSED: Cannot delete top-level runs directory: ${abs_path}"
        log_info "Delete specific subdirectories instead (e.g., runs/phase2/variant/seed_N)"
        return 1
    fi

    # Generate trash name with timestamp
    local timestamp
    timestamp="$(date +%Y%m%d-%H%M%S)"
    local trash_name="${PROJECT_TAG}--${basename_target}--${timestamp}"
    local trash_path="${TRASH_DIR}/${trash_name}"

    # Move to trash
    mv "$abs_path" "$trash_path"

    # Save metadata for restore
    local meta="${trash_path}.meta"
    cat > "$meta" << EOF
original_path=${abs_path}
deleted_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
deleted_by=safe-delete.sh
EOF

    local size
    size=$(du -sh "$trash_path" 2>/dev/null | awk '{print $1}')
    log_ok "Moved to trash: ${target} → ~/.Trash/${trash_name} (${size})"
}

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "$1" in
    --list|-l)
        cmd_list
        ;;
    --restore|-r)
        if [ $# -lt 2 ]; then
            log_error "Usage: $0 --restore <trash-name>"
            exit 1
        fi
        cmd_restore "$2"
        ;;
    --help|-h)
        usage
        ;;
    *)
        # Delete one or more paths
        echo -e "${BOLD}Safe Delete — Moving to Trash${NC}"
        for target in "$@"; do
            safe_delete "$target"
        done
        echo ""
        echo -e "Restore with: ${BLUE}$0 --list${NC} then ${BLUE}$0 --restore <name>${NC}"
        ;;
esac
