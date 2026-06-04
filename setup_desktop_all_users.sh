#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_desktop_all_users.sh
#
# Puts an ACORN desktop icon on every user's Desktop and updates the
# system application menu entry.
#
# Run with:   sudo bash /home/vnw/acorn/setup_desktop_all_users.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo" >&2; exit 1
fi

ACORN_LAUNCH="/home/vnw/acorn/acorn2.sh"
ACORN_ICON="/home/vnw/acorn/src/acorn/gui/acorn.png"
DESKTOP_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=ACORN
Comment=Microscopy image analysis with AI-assisted annotation
Exec=bash ${ACORN_LAUNCH}
Icon=${ACORN_ICON}
Terminal=false
Categories=Science;Education;
StartupNotify=true
"

BOLD="\033[1m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RESET="\033[0m"
info()    { echo -e "${BOLD}[ACORN]${RESET} $*"; }
success() { echo -e "${GREEN}[ACORN]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[ACORN]${RESET} $*"; }

# ── 1. Verify launch script exists ────────────────────────────────────────────
if [ ! -f "$ACORN_LAUNCH" ]; then
    echo "ERROR: $ACORN_LAUNCH not found." >&2; exit 1
fi
if [ ! -f "$ACORN_ICON" ]; then
    warn "Icon not found at $ACORN_ICON — desktop entries will still work without it."
fi

# ── 2. Update system application menu entry ───────────────────────────────────
info "Updating /usr/share/applications/ACORN.desktop ..."
cat > /usr/share/applications/ACORN.desktop <<EOF
${DESKTOP_CONTENT}
EOF
chmod 644 /usr/share/applications/ACORN.desktop
success "System app menu updated."

# ── 3. Update /etc/skel so future users get the icon automatically ─────────────
info "Updating /etc/skel ..."
mkdir -p /etc/skel/Desktop
cat > /etc/skel/Desktop/ACORN.desktop <<EOF
${DESKTOP_CONTENT}
EOF
chmod 644 /etc/skel/Desktop/ACORN.desktop
success "skel updated — new users will get the icon automatically."

# ── 4. Put icon on every existing user's Desktop ──────────────────────────────
info "Adding icon to all user desktops ..."
SKIPPED=0
ADDED=0

for homedir in /home/*/; do
    user=$(basename "$homedir")
    # Skip system-like accounts without a real shell
    shell=$(getent passwd "$user" 2>/dev/null | cut -d: -f7)
    if [[ "$shell" == */nologin || "$shell" == */false || -z "$shell" ]]; then
        continue
    fi

    desktop_dir="${homedir}Desktop"
    dest="${desktop_dir}/ACORN.desktop"

    mkdir -p "$desktop_dir"

    cat > "$dest" <<EOF
${DESKTOP_CONTENT}
EOF
    chmod 755 "$dest"
    chown "${user}:" "$dest" 2>/dev/null || true
    chown "${user}:" "$desktop_dir" 2>/dev/null || true

    # Mark as trusted (suppresses "Untrusted launcher" dialog on GNOME/XFCE)
    # gio sets xattr metadata; silently skip if gio is unavailable
    sudo -u "$user" gio set "$dest" metadata::trusted true 2>/dev/null || true

    ADDED=$((ADDED + 1))
done

success "Desktop icon added for $ADDED user(s).  $SKIPPED skipped (no shell)."

# ── 5. Update desktop database ─────────────────────────────────────────────────
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database /usr/share/applications
    success "Desktop database refreshed."
fi

echo ""
echo -e "${GREEN}${BOLD}Done. ACORN is now on every user's desktop.${RESET}"
echo -e "Launch script: ${BOLD}${ACORN_LAUNCH}${RESET}"
echo ""
