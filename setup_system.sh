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
DST="/opt/acorn"
VENV="$DST/.venv"
OWNER="vnw"                          # user who owns /opt/acorn and can deploy
# System Python — prefers the shared conda install; falls back to any python3 >= 3.10
PYTHON="${ACORN_PYTHON:-$(command -v /opt/conda/bin/python3 2>/dev/null || command -v python3)}"
UV="$(command -v uv || echo /home/vnw/.local/bin/uv)"
SAM3_SRC="/home/vnw/repos/sam3"      # editable sam3 dev source — install as regular package

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
mkdir -p "$DST"/{src,models/micro_sam,models/yolo}
chown -R "$OWNER":users "$DST"
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
rsync -a "$SRC/pyproject.toml"      "$DST/pyproject.toml"
rsync -a "$SRC/README.md"           "$DST/README.md"
rsync -a "$SRC/download_models.py"  "$DST/download_models.py"
rsync -a "$SRC/deploy.sh"           "$DST/deploy.sh"
rsync -a "$SRC/install.sh"          "$DST/install.sh"
chmod a+rX -R "$DST/src"
chown -R "$OWNER":users "$DST"

# ── 3. Create virtual environment ─────────────────────────────────────────────
info "Creating Python environment at $VENV..."
if [ -d "$VENV" ]; then
    warn "  Existing venv found — removing and rebuilding."
    rm -rf "$VENV"
fi
"$UV" venv "$VENV" --python "$PYTHON"
chown -R "$OWNER":users "$VENV"
chmod -R a+rX "$VENV"

VENV_PY="$VENV/bin/python"

# ── 4. Install packages ───────────────────────────────────────────────────────
info "Installing ACORN and dependencies..."
# setuptools provides pkg_resources, which some packages (e.g. sam3) still use
"$UV" pip install --python "$VENV_PY" setuptools --quiet
"$UV" pip install --python "$VENV_PY" -e "$DST[gui,mrc]" --quiet

info "Installing AI tools (SAM, YOLO, UNet)..."
# segment-anything (Meta SAM1) and micro-sam are not on PyPI — install from GitHub.
# segment-anything is required by usam_predictor; micro-sam provides fine-tuned checkpoints.
"$UV" pip install --python "$VENV_PY" \
    "git+https://github.com/facebookresearch/segment-anything.git" \
    "git+https://github.com/computational-cell-analytics/micro-sam.git" \
    "ultralytics>=8.0" "segmentation-models-pytorch>=0.3" --quiet \
    || warn "Some AI packages failed — optional features may be limited."

# Install sam3 from source as a regular (non-editable) package.
# Use --no-deps to skip sam3's numpy==1.26 pin (incompatible with Python 3.13);
# numpy is already installed by the acorn[gui,mrc] step above.
if [ -d "$SAM3_SRC" ]; then
    info "Installing sam3 from $SAM3_SRC..."
    "$UV" pip install --python "$VENV_PY" "$SAM3_SRC" --no-deps --quiet \
        && "$UV" pip install --python "$VENV_PY" \
            "timm>=1.0.17" "tqdm" "ftfy==6.1.1" "regex" \
            "iopath>=0.1.10" "typing_extensions" "huggingface_hub" --quiet \
        || warn "sam3 install failed — SAM3 backend will be unavailable."
else
    warn "sam3 source not found at $SAM3_SRC — SAM3 backend will be unavailable."
fi

# Ensure permissions are open after install
chmod -R a+rX "$VENV"
chown -R "$OWNER":users "$VENV"

# ── 5. System-wide environment variables ──────────────────────────────────────
info "Writing /etc/profile.d/acorn.sh..."
cat > /etc/profile.d/acorn.sh << 'EOF'
# ACORN shared model cache — set for all users
if [ -d /opt/acorn/models ]; then
    export MICROSAM_CACHEDIR=/opt/acorn/models/micro_sam
    export ACORN_MODELS_DIR=/opt/acorn/models
fi
EOF
chmod 644 /etc/profile.d/acorn.sh

# ── 6. CLI wrappers in /usr/local/bin ─────────────────────────────────────────
info "Creating CLI commands: acorn, acorn-gui..."

cat > /usr/local/bin/acorn << EOF
#!/usr/bin/env bash
export MICROSAM_CACHEDIR=/opt/acorn/models/micro_sam
export ACORN_MODELS_DIR=/opt/acorn/models
source /opt/acorn/.venv/bin/activate
exec /opt/acorn/.venv/bin/acorn "\$@"
EOF

cat > /usr/local/bin/acorn-gui << EOF
#!/usr/bin/env bash
export MICROSAM_CACHEDIR=/opt/acorn/models/micro_sam
export ACORN_MODELS_DIR=/opt/acorn/models
source /opt/acorn/.venv/bin/activate
exec /opt/acorn/.venv/bin/acorn-gui "\$@"
EOF

chmod 755 /usr/local/bin/acorn /usr/local/bin/acorn-gui

# ── 7. Desktop entry (ThinLinc / GNOME / KDE) ─────────────────────────────────
info "Creating desktop entry for all users..."
ICON="/opt/acorn/src/acorn/gui/acorn.png"

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

# Run as the owner so files are owned by vnw, not root
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
echo -e "    cd /home/$OWNER/cryoem-tools && ${BOLD}bash deploy.sh${RESET}"
echo ""
