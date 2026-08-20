"""
Three button roles, so a button's colour says what it does.

Buttons were styled inline at 42 sites in two different "this is the main action"
colours — the ORNL green and a saturated blue — with no rule about which meant
what. The Export panel showed all three treatments at once: a green Save, a grey
Save Raw TIFF, and a blue Export ROI Masks, which reads as three unrelated kinds
of action rather than one primary and two alternatives.

    primary   the action the panel exists for  — one per panel
    secondary a real action, but not the point of the panel
    danger    deletes or overwrites something

Secondary is deliberately not a colour. Two saturated colours competing in one
panel is what made the old layout hard to read; emphasis comes from the border
and weight instead, which also keeps the accent meaningful when it does appear.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QAbstractButton

from acorn.render import palette as _PAL

_GREEN       = "#00703C"      # ORNL green — the brand, and the primary action
_GREEN_HOVER = "#008A4B"
_SURFACE     = "#252525"
_SURFACE_HI  = "#363636"
_BORDER      = "#454545"
_TEXT        = "#e0e0e0"
_DANGER      = "#B3452F"
_DANGER_HI   = "#C7543D"

_PRIMARY_QSS = f"""
QPushButton {{
    background: {_GREEN}; color: #ffffff; font-weight: 600;
    border: 1px solid {_GREEN}; border-radius: 4px; padding: 5px 12px;
}}
QPushButton:hover   {{ background: {_GREEN_HOVER}; border-color: {_GREEN_HOVER}; }}
QPushButton:pressed {{ background: {_GREEN}; }}
QPushButton:disabled {{
    background: {_SURFACE}; color: #6f6f6f; border-color: {_BORDER};
}}
"""

_SECONDARY_QSS = f"""
QPushButton {{
    background: {_SURFACE_HI}; color: {_TEXT}; font-weight: 600;
    border: 1px solid {_BORDER}; border-radius: 4px; padding: 5px 12px;
}}
QPushButton:hover   {{ background: {_BORDER}; border-color: {_PAL.ACCEPTED}; }}
QPushButton:pressed {{ background: {_SURFACE}; }}
QPushButton:disabled {{ background: {_SURFACE}; color: #6f6f6f; }}
"""

_DANGER_QSS = f"""
QPushButton {{
    background: {_DANGER}; color: #ffffff; font-weight: 600;
    border: 1px solid {_DANGER}; border-radius: 4px; padding: 5px 12px;
}}
QPushButton:hover   {{ background: {_DANGER_HI}; border-color: {_DANGER_HI}; }}
QPushButton:disabled {{ background: {_SURFACE}; color: #6f6f6f; border-color: {_BORDER}; }}
"""

_QSS = {"primary": _PRIMARY_QSS, "secondary": _SECONDARY_QSS, "danger": _DANGER_QSS}


def style(button: QAbstractButton, role: str = "secondary") -> QAbstractButton:
    """Apply a role to *button* and return it, so it can be used inline."""
    button.setStyleSheet(_QSS.get(role, _SECONDARY_QSS))
    return button


def primary(button: QAbstractButton) -> QAbstractButton:
    """The action this panel exists for. Use once per panel."""
    return style(button, "primary")


def secondary(button: QAbstractButton) -> QAbstractButton:
    """A real action that is not the point of the panel."""
    return style(button, "secondary")


def danger(button: QAbstractButton) -> QAbstractButton:
    """Deletes or overwrites something."""
    return style(button, "danger")
