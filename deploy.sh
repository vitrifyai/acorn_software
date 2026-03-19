#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh  —  push the dev copy to the shared /opt/acorn installation.
#
# Run this from /home/vnw/cryoem-tools whenever you want to release an update.
# No sudo needed after the initial setup_system.sh has been run.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="/opt/acorn"
VENV="$DST/.venv"
VENV_PY="$VENV/bin/python"

BOLD="\033[1m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RESET="\033[0m"
info()    { echo -e "${BOLD}[deploy]${RESET} $*"; }
success() { echo -e "${GREEN}[deploy]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[deploy]${RESET} $*"; }

if [ ! -d "$DST" ]; then
    echo "ERROR: $DST does not exist.  Run setup_system.sh first." >&2
    exit 1
fi

# ── 1. Sync source code ───────────────────────────────────────────────────────
info "Syncing source → $DST"
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
rsync -a "$SRC/install.sh"          "$DST/install.sh"

# ── 2. Reinstall if pyproject.toml changed ────────────────────────────────────
STAMP="$DST/.last_pyproject_hash"
NEW_HASH=$(sha256sum "$DST/pyproject.toml" | awk '{print $1}')
OLD_HASH=$(cat "$STAMP" 2>/dev/null || echo "")

if [ "$NEW_HASH" != "$OLD_HASH" ]; then
    info "pyproject.toml changed — reinstalling dependencies..."
    uv pip install --python "$VENV_PY" -e "$DST[gui,mrc]" --quiet
    echo "$NEW_HASH" > "$STAMP"
    success "Dependencies updated."
else
    info "Dependencies unchanged — skipping reinstall."
fi

# ── 3. Fix permissions ────────────────────────────────────────────────────────
chmod -R a+rX "$DST/src"

success "Deploy complete.  /opt/acorn is up to date."
echo -e "  Users can launch with: ${BOLD}acorn-gui${RESET}  or the Desktop icon."
