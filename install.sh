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
UV="${UV:-}"
if [ -z "$UV" ]; then
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ] && "$candidate" --version >/dev/null 2>&1; then
            UV="$candidate"
            break
        fi
    done
fi

if [ -z "$UV" ]; then
    info "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    UV="$(command -v uv 2>/dev/null || true)"
    [ -n "$UV" ] || die "uv install failed.  Please install it manually:\n  curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

UV_VERSION=$("$UV" --version)
info "Using $UV_VERSION"

# ── 2. Create (or update) the virtual environment ────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_VERSION="${ACORN_PYTHON_VERSION:-3.12}"
INSTALL_EXTRA="${ACORN_INSTALL_EXTRA:-full}"

if [ -d "$VENV_DIR" ]; then
    if "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        warn "Existing Python 3.12+ environment found — updating it."
    else
        warn "Existing environment uses Python older than 3.12 — rebuilding it."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating isolated Python $PYTHON_VERSION environment..."
    "$UV" python install "$PYTHON_VERSION"
    "$UV" venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

# ── 3. Install ACORN full application ─────────────────────────────────────────
info "Installing ACORN [$INSTALL_EXTRA] from uv.lock..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV" sync --frozen --extra "$INSTALL_EXTRA"
success "ACORN [$INSTALL_EXTRA] installation complete from locked dependencies."

# ── 4. Pre-download model checkpoints ────────────────────────────────────────
echo ""
info "Downloading recommended AI model checkpoints..."
info "  SAM EM organelles checkpoint  (~375 MB)"
info "  YOLO nano segmentation model  (~6 MB)"
info "  (This only happens once.  Models are saved to ~/.cache and ~/.acorn)"
info "  To download more models later, run:  python download_models.py"
echo ""
"$VENV_PYTHON" "$SCRIPT_DIR/download_models.py" --preset recommended \
    || warn "Some models failed to download.  Run 'python download_models.py' after connecting to the internet."

# ── 5. Write the launch script ────────────────────────────────────────────────
LAUNCHER="$SCRIPT_DIR/acorn.sh"
cat > "$LAUNCHER" << EOF
#!/usr/bin/env bash
# Launch ACORN.
source "$VENV_DIR/bin/activate"
exec acorn-gui "\$@"
EOF
chmod +x "$LAUNCHER"

# ── 6. Create desktop shortcut (Linux) ────────────────────────────────────────
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

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}========================================${RESET}"
echo -e "${GREEN}${BOLD}   Installation complete!               ${RESET}"
echo -e "${GREEN}${BOLD}========================================${RESET}"
echo ""
echo -e "  To launch ACORN:"
echo -e "    ${BOLD}Double-click ACORN on your Desktop${RESET}"
echo -e "    ${BOLD}or run:  ./acorn.sh${RESET}"
echo ""
