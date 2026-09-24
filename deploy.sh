#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh  —  push the dev copy to the shared /opt/acorn installation.
#
# Run this from the ACORN source checkout whenever you want to release an update.
# No sudo needed after the initial setup_system.sh has been run.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${ACORN_PREFIX:-/opt/acorn}"
VENV="$DST/.venv"
VENV_PY="$VENV/bin/python"
INSTALL_EXTRA="${ACORN_INSTALL_EXTRA:-full}"
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
    echo "ERROR: uv is not installed or is not runnable. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

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
rsync -a "$SRC/install.sh"          "$DST/install.sh"

# ── 2. Reinstall if dependency metadata changed ───────────────────────────────
STAMP="$DST/.last_pyproject_hash"
NEW_HASH=$(
    {
        sha256sum "$DST/pyproject.toml"
        sha256sum "$DST/uv.lock"
        find "$DST/packages" -name pyproject.toml -type f -print0 | sort -z | xargs -0 sha256sum
    } | sha256sum | awk '{print $1}'
)
OLD_HASH=$(cat "$STAMP" 2>/dev/null || echo "")

if [ "$NEW_HASH" != "$OLD_HASH" ]; then
    info "Dependency metadata changed — syncing locked dependencies..."
    cd "$DST"
    UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --frozen --extra "$INSTALL_EXTRA" --quiet
    echo "$NEW_HASH" > "$STAMP"
    success "Dependencies updated."
else
    info "Dependencies unchanged — skipping reinstall."
fi

# ── 3. Fix permissions ────────────────────────────────────────────────────────
chmod -R a+rX "$DST/src"
chmod -R a+rX "$DST/packages"

success "Deploy complete.  /opt/acorn is up to date."
echo -e "  Users can launch with: ${BOLD}acorn-gui${RESET}  or the Desktop icon."
