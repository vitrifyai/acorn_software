#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ACORN updater — run this after 'git pull' to pick up any new dependencies.
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

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

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

# Ensure uv is available
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}   ACORN — Updating dependencies        ${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

info "Updating ACORN core..."
uv pip install --python "$VENV_PYTHON" -e ".[gui,mrc]" --quiet
success "Core updated."

info "Checking SAM3..."
if ! "$VENV_PYTHON" -c "import sam3" 2>/dev/null; then
    info "  SAM3 not found — installing..."
    uv pip install --python "$VENV_PYTHON" "git+https://github.com/facebookresearch/sam3.git" \
        && success "  SAM3 installed." \
        || warn "  SAM3 install failed. Run: uv pip install --python .venv/bin/python git+https://github.com/facebookresearch/sam3.git"
else
    success "  SAM3 already installed."
fi

info "Checking micro-SAM..."
if ! "$VENV_PYTHON" -c "import micro_sam" 2>/dev/null; then
    info "  micro-SAM not found — installing..."
    uv pip install --python "$VENV_PYTHON" "git+https://github.com/computational-cell-analytics/micro-sam.git" \
        && success "  micro-SAM installed." \
        || warn "  micro-SAM install failed. Run: uv pip install --python .venv/bin/python git+https://github.com/computational-cell-analytics/micro-sam.git"
else
    success "  micro-SAM already installed."
fi

info "Checking YOLO..."
if ! "$VENV_PYTHON" -c "import ultralytics" 2>/dev/null; then
    uv pip install --python "$VENV_PYTHON" "ultralytics>=8.0" \
        && success "  YOLO installed." \
        || warn "  YOLO install failed."
else
    success "  YOLO already installed."
fi

info "Checking UNet..."
if ! "$VENV_PYTHON" -c "import segmentation_models_pytorch" 2>/dev/null; then
    uv pip install --python "$VENV_PYTHON" "segmentation-models-pytorch>=0.3" \
        && success "  UNet installed." \
        || warn "  UNet install failed."
else
    success "  UNet already installed."
fi

info "Checking analysis tools (opencv, pandas)..."
if ! "$VENV_PYTHON" -c "import cv2; import pandas" 2>/dev/null; then
    uv pip install --python "$VENV_PYTHON" "opencv-python>=4.8" "pandas>=1.5" \
        && success "  Analysis tools installed." \
        || warn "  Analysis tools install failed."
else
    success "  Analysis tools already installed."
fi

info "Checking CLU assistant (anthropic, openai)..."
if ! "$VENV_PYTHON" -c "import anthropic" 2>/dev/null; then
    uv pip install --python "$VENV_PYTHON" "anthropic>=0.52" "openai>=1.0" \
        && success "  CLU assistant installed." \
        || warn "  CLU assistant install failed."
else
    success "  CLU assistant already installed."
fi

echo ""
echo -e "${GREEN}${BOLD}Update complete. Launch with: ./acorn.sh${RESET}"
echo ""
