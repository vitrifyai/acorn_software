#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ACORN updater - run this after pulling new code to sync locked dependencies.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RESET="\033[0m"

info()    { echo -e "${BOLD}[ACORN]${RESET} $*"; }
success() { echo -e "${GREEN}[ACORN]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[ACORN]${RESET} $*"; }

VENV_DIR="${ACORN_VENV_DIR:-$SCRIPT_DIR/.venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
INSTALL_EXTRA="${ACORN_INSTALL_EXTRA:-full}"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "No virtual environment found — run bash install.sh first."
    exit 1
fi

# Ensure git is available
if ! command -v git &>/dev/null; then
    echo "git is not installed. Please ask your system administrator to run:"
    echo "  sudo apt install git"
    echo "then re-run this script."
    exit 1
fi

find_uv() {
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
        [ -n "$candidate" ] || continue
        if [ -x "$candidate" ] && "$candidate" --version >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

UV="$(find_uv || true)"
if [ -z "$UV" ]; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$(find_uv || true)"
fi
if [ -z "$UV" ]; then
    echo "uv could not be installed or found on PATH." >&2
    exit 1
fi

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}   ACORN - Updating dependencies        ${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

info "Syncing locked environment: .[$INSTALL_EXTRA]"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV" sync --frozen --extra "$INSTALL_EXTRA"
success "Environment synced."

if [ -f "$SCRIPT_DIR/download_models.py" ]; then
    info "Checking recommended model cache..."
    "$VENV_PYTHON" "$SCRIPT_DIR/download_models.py" --preset recommended \
        || warn "Model download skipped or incomplete; ACORN can still start."
fi

echo ""
echo -e "${GREEN}${BOLD}Update complete. Launch with: ./acorn2.sh${RESET}"
echo ""
