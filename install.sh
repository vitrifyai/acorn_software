#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ACORN installer
# Run this once to set up everything.  No prior Python knowledge needed.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

info()    { echo -e "${BOLD}[ACORN]${RESET} $*"; }
success() { echo -e "${GREEN}[ACORN]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[ACORN]${RESET} $*"; }
die()     { echo -e "${RED}[ACORN] ERROR:${RESET} $*" >&2; exit 1; }

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}   ACORN — Microscopy Analysis Suite   ${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

# ── 1. Ensure uv is available ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv &>/dev/null || die "uv install failed.  Please install it manually:\n  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

UV_VERSION=$(uv --version)
info "Using $UV_VERSION"

# ── 2. Create (or update) the virtual environment ────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    warn "Existing environment found — updating it."
else
    info "Creating isolated Python environment..."
    uv venv "$VENV_DIR" --python 3.10 2>/dev/null \
        || uv venv "$VENV_DIR"   # fall back to any available Python >= 3.10
fi

VENV_PYTHON="$VENV_DIR/bin/python"

# ── 3. Install ACORN core + GUI ───────────────────────────────────────────────
info "Installing ACORN (this may take a couple of minutes)..."
uv pip install --python "$VENV_PYTHON" -e ".[gui,mrc]"
success "Core installation complete."

# ── 4. Install AI annotation tools ───────────────────────────────────────────
echo ""
info "Installing AI-assisted annotation tools (SAM, YOLO, UNet)..."

info "  SAM — Segment Anything Model..."
uv pip install --python "$VENV_PYTHON" "micro-sam>=1.7" "sam3>=0.1" \
    || warn "  SAM install had issues — SAM tab may be limited."

info "  YOLO — object detection..."
uv pip install --python "$VENV_PYTHON" "ultralytics>=8.0" \
    || warn "  YOLO install failed — YOLO tab will be unavailable."

info "  UNet — semantic segmentation..."
uv pip install --python "$VENV_PYTHON" "segmentation-models-pytorch>=0.3" \
    || warn "  UNet install failed — UNet tab will be unavailable."

success "AI tools installed."

# ── 5. Pre-download model checkpoints ────────────────────────────────────────
echo ""
info "Downloading recommended AI model checkpoints..."
info "  SAM EM organelles checkpoint  (~375 MB)"
info "  YOLO nano segmentation model  (~6 MB)"
info "  (This only happens once.  Models are saved to ~/.cache and ~/.acorn)"
info "  To download more models later, run:  python download_models.py"
echo ""
"$VENV_PYTHON" "$SCRIPT_DIR/download_models.py" --preset recommended \
    || warn "Some models failed to download.  Run 'python download_models.py' after connecting to the internet."

# ── 6. Write the launch script ────────────────────────────────────────────────
LAUNCHER="$SCRIPT_DIR/acorn.sh"
cat > "$LAUNCHER" << EOF
#!/usr/bin/env bash
# Launch ACORN.
source "$VENV_DIR/bin/activate"
exec acorn-gui "\$@"
EOF
chmod +x "$LAUNCHER"

# ── 7. Create desktop shortcut (Linux) ────────────────────────────────────────
ICON_PATH="$SCRIPT_DIR/src/acorn/gui/acorn.png"
DESKTOP_DIR="$HOME/Desktop"
DESKTOP_FILE="$DESKTOP_DIR/ACORN.desktop"

if [ -d "$DESKTOP_DIR" ]; then
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ACORN
Comment=Microscopy image analysis and annotation
Exec=$LAUNCHER
Icon=$ICON_PATH
Terminal=false
Categories=Science;Education;
StartupNotify=true
EOF
    chmod +x "$DESKTOP_FILE"
    gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
    success "Desktop shortcut created."
fi

# ── 7. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}========================================${RESET}"
echo -e "${GREEN}${BOLD}   Installation complete!               ${RESET}"
echo -e "${GREEN}${BOLD}========================================${RESET}"
echo ""
echo -e "  To launch ACORN:"
echo -e "    ${BOLD}Double-click ACORN on your Desktop${RESET}"
echo -e "    ${BOLD}or run:  ./acorn.sh${RESET}"
echo ""
