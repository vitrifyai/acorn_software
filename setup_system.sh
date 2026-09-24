#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_system.sh  —  one-time system-wide installation of ACORN to /opt/acorn.
#
# Must be run with sudo:
#     sudo bash setup_system.sh
#
# After this runs, use deploy.sh (no sudo) to push updates.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run this with sudo:  sudo bash setup_system.sh" >&2
    exit 1
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${ACORN_PREFIX:-/opt/acorn}"
VENV="$DST/.venv"
OWNER="${ACORN_OWNER:-${SUDO_USER:-}}"
if [ -z "$OWNER" ] || [ "$OWNER" = "root" ]; then
    echo "ERROR: set ACORN_OWNER to the non-root user who should own $DST" >&2
    exit 1
fi
if ! id "$OWNER" >/dev/null 2>&1; then
    echo "ERROR: ACORN_OWNER user '$OWNER' does not exist" >&2
    exit 1
fi
GROUP="${ACORN_GROUP:-$(id -gn "$OWNER")}"
OWNER_HOME="$(getent passwd "$OWNER" | cut -d: -f6)"
INSTALL_EXTRA="${ACORN_INSTALL_EXTRA:-full}"
# System Python — prefers an explicit ACORN_PYTHON, otherwise asks uv for 3.12.
PYTHON="${ACORN_PYTHON:-3.12}"
UV="${UV:-}"
if [ -z "$UV" ]; then
    for candidate in "$OWNER_HOME/.local/bin/uv" "$OWNER_HOME/.cargo/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ] && "$candidate" --version >/dev/null 2>&1; then
            UV="$candidate"
            break
        fi
    done
fi
if [ -z "$UV" ]; then
    echo "ERROR: uv is not installed or is not runnable. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
SAM3_SRC="${ACORN_SAM3_SRC:-}"

BOLD="\033[1m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RESET="\033[0m"
info()    { echo -e "${BOLD}[setup]${RESET} $*"; }
success() { echo -e "${GREEN}[setup]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[setup]${RESET} $*"; }

echo ""
echo -e "${BOLD}======================================================${RESET}"
echo -e "${BOLD}   ACORN system-wide setup → $DST${RESET}"
echo -e "${BOLD}======================================================${RESET}"
echo ""

# ── 1. Create directory structure ─────────────────────────────────────────────
info "Creating $DST..."
mkdir -p "$DST"/{src,packages,models/micro_sam,models/yolo}
chown -R "$OWNER":"$GROUP" "$DST"
chmod 755 "$DST"
chmod -R 755 "$DST/models"   # readable by all, writable by owner

# ── 2. Copy source files ──────────────────────────────────────────────────────
info "Copying source files..."
rsync -a --delete \
    --exclude=".venv" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".git" \
    --exclude="models" \
    "$SRC/src/"            "$DST/src/"
rsync -a --delete \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    "$SRC/packages/"       "$DST/packages/"
rsync -a "$SRC/pyproject.toml"      "$DST/pyproject.toml"
rsync -a "$SRC/.python-version"     "$DST/.python-version"
rsync -a "$SRC/uv.lock"             "$DST/uv.lock"
rsync -a "$SRC/README.md"           "$DST/README.md"
rsync -a "$SRC/QUICKSTART.md"       "$DST/QUICKSTART.md"
rsync -a "$SRC/download_models.py"  "$DST/download_models.py"
rsync -a "$SRC/deploy.sh"           "$DST/deploy.sh"
rsync -a "$SRC/install.sh"          "$DST/install.sh"
chmod a+rX -R "$DST/src"
chmod a+rX -R "$DST/packages"
chown -R "$OWNER":"$GROUP" "$DST"

# ── 3. Create virtual environment ─────────────────────────────────────────────
info "Creating Python environment at $VENV..."
if [ -d "$VENV" ]; then
    warn "  Existing venv found — removing and rebuilding."
    rm -rf "$VENV"
fi
"$UV" python install "$PYTHON"
"$UV" venv "$VENV" --python "$PYTHON"
chown -R "$OWNER":"$GROUP" "$VENV"
chmod -R a+rX "$VENV"

VENV_PY="$VENV/bin/python"
"$VENV_PY" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("ACORN requires Python 3.12 or newer for the full install")
PY

# ── 4. Install packages ───────────────────────────────────────────────────────
info "Installing ACORN [$INSTALL_EXTRA] from locked dependencies..."
cd "$DST"
UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --frozen --extra "$INSTALL_EXTRA" --quiet

if [ -n "$SAM3_SRC" ] && [ -d "$SAM3_SRC" ]; then
    info "Installing sam3 from $SAM3_SRC..."
    "$UV" pip install --python "$VENV_PY" "$SAM3_SRC" --no-deps --quiet \
        || warn "sam3 install failed — SAM3 backend will be unavailable."
fi

# Ensure permissions are open after install
chmod -R a+rX "$VENV"
chown -R "$OWNER":"$GROUP" "$VENV"

# ── 5. System-wide environment variables ──────────────────────────────────────
info "Writing /etc/profile.d/acorn.sh..."
cat > /etc/profile.d/acorn.sh << EOF
# ACORN shared model cache — set for all users
if [ -d "$DST/models" ]; then
    export MICROSAM_CACHEDIR="$DST/models/micro_sam"
    export ACORN_MODELS_DIR="$DST/models"
fi
EOF
chmod 644 /etc/profile.d/acorn.sh

# ── 6. CLI wrappers in /usr/local/bin ─────────────────────────────────────────
info "Creating CLI commands: acorn, acorn-gui..."

cat > /usr/local/bin/acorn << EOF
#!/usr/bin/env bash
export MICROSAM_CACHEDIR="$DST/models/micro_sam"
export ACORN_MODELS_DIR="$DST/models"
source "$DST/.venv/bin/activate"
exec "$DST/.venv/bin/acorn" "\$@"
EOF

cat > /usr/local/bin/acorn-gui << EOF
#!/usr/bin/env bash
export MICROSAM_CACHEDIR="$DST/models/micro_sam"
export ACORN_MODELS_DIR="$DST/models"
source "$DST/.venv/bin/activate"
exec "$DST/.venv/bin/acorn-gui" "\$@"
EOF

chmod 755 /usr/local/bin/acorn /usr/local/bin/acorn-gui

# ── 7. Desktop entry (ThinLinc / GNOME / KDE) ─────────────────────────────────
info "Creating desktop entry for all users..."
ICON="$DST/src/acorn/gui/acorn.png"

cat > /usr/share/applications/acorn.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ACORN
GenericName=Microscopy Analysis
Comment=Annotate, Curate, Observe, Review, Navigate — cryo-EM image analysis
Exec=/usr/local/bin/acorn-gui %f
Icon=$ICON
Terminal=false
Categories=Science;Education;
MimeType=image/tiff;
Keywords=microscopy;cryo-em;annotation;segmentation;
StartupNotify=true
EOF

chmod 644 /usr/share/applications/acorn.desktop

# Update desktop database so it appears immediately
update-desktop-database /usr/share/applications/ 2>/dev/null || true

# ── 8. Pre-download shared models ─────────────────────────────────────────────
info "Downloading shared model checkpoints to $DST/models/..."
info "  (All users will share these — no per-user downloads needed)"
echo ""

# Run as the owner so files are not owned by root
sudo -u "$OWNER" \
    env MICROSAM_CACHEDIR="$DST/models/micro_sam" \
        ACORN_MODELS_DIR="$DST/models" \
    "$VENV_PY" "$DST/download_models.py" --preset recommended \
    || warn "Model download failed — run 'python download_models.py' as $OWNER to retry."

chmod -R a+rX "$DST/models"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}======================================================${RESET}"
echo -e "${GREEN}${BOLD}   System setup complete!                             ${RESET}"
echo -e "${GREEN}${BOLD}======================================================${RESET}"
echo ""
echo "  Every user on this machine can now:"
echo -e "    Run the GUI:  ${BOLD}acorn-gui${RESET}"
echo -e "    Run the CLI:  ${BOLD}acorn${RESET}"
echo -e "    Or click the ${BOLD}ACORN${RESET} icon in the application menu"
echo ""
echo "  To push future updates (no sudo needed):"
echo -e "    cd $SRC && ${BOLD}bash deploy.sh${RESET}"
echo ""
