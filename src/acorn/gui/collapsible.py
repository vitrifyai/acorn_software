"""
Folding the group boxes in a control panel.

The panels grew past what a screen holds. Measured at a 1400x900 window, with
every group expanded:

    Annotate  2181px    Segment  1126px    Train     649px
    Export    1516px    Measure   923px    Contrast  615px

against a viewport of roughly 630-750px. Annotate is three and a half screens of
scrolling to reach a button, and most of what you scroll past belongs to a tool
you are not using at that moment.

Rather than redesign six panels, this makes any existing QGroupBox foldable in
place: click the title to collapse it. Panels keep constructing plain group
boxes, and `apply_to_panel` folds them afterwards, so nothing has to be rewritten
and a new group gets the behaviour for free.

Which sections start folded is remembered per panel, so the arrangement someone
settles on survives a restart.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QWidget

_ARROW_OPEN = "▾"      # ▾
_ARROW_SHUT = "▸"      # ▸

_STYLE = """
QGroupBox::title {
    subcontrol-origin: margin;
    left: 6px;
    padding: 0 4px;
}
QGroupBox[folded="true"] {
    border: 1px solid #363636;
    border-radius: 4px;
    margin-top: 6px;
}
"""


def _title_without_arrow(text: str) -> str:
    return text.lstrip(_ARROW_OPEN + _ARROW_SHUT + " ").strip()


def _direct_children(group: QGroupBox) -> list[QWidget]:
    """Widgets the group owns directly. Hiding these hides their descendants too."""
    return [c for c in group.findChildren(QWidget) if c.parentWidget() is group]


def set_folded(group: QGroupBox, folded: bool) -> None:
    """
    Fold or unfold *group*, hiding its contents and shrinking it to the title.

    Only direct children are touched, and their visibility is recorded before
    folding so unfolding restores what was there rather than revealing
    everything. Panels hide things on purpose - the inactive pages of a stacked
    widget, a control that only applies to the current method - and showing all
    descendants indiscriminately made two panels TALLER when folded groups were
    reopened.
    """
    children = _direct_children(group)
    if folded:
        group.setProperty("_prev_visible",
                          [c.isVisibleTo(group) for c in children])
        for child in children:
            child.setVisible(False)
    else:
        prev = group.property("_prev_visible")
        if isinstance(prev, list) and len(prev) == len(children):
            for child, was_visible in zip(children, prev):
                child.setVisible(bool(was_visible))
        # With no record there is nothing to restore, and forcing everything
        # visible would reveal controls the panel deliberately hid. Unfolding
        # something that was never folded should change nothing.
    title = _title_without_arrow(group.title())
    group.setTitle(f"{_ARROW_SHUT} {title}" if folded else f"{_ARROW_OPEN} {title}")
    group.setProperty("folded", "true" if folded else "false")
    group.setFlat(folded)
    if folded:
        # Just the title bar. Without this the empty box keeps its full height.
        group.setMaximumHeight(group.fontMetrics().height() + 14)
    else:
        group.setMaximumHeight(16777215)
    group.style().unpolish(group)
    group.style().polish(group)


def is_folded(group: QGroupBox) -> bool:
    return group.property("folded") == "true"


def make_collapsible(group: QGroupBox, folded: bool = False, on_toggle=None) -> QGroupBox:
    """
    Let *group* be folded by clicking its title.

    Uses QGroupBox's own checkable title as the hit target, because it is already
    positioned and styled correctly, then hides the contents rather than merely
    disabling them, which is what Qt does by default and does not save any space.
    """
    group.setCheckable(True)
    group.setChecked(not folded)
    group.setStyleSheet(group.styleSheet() + _STYLE)
    group.setCursor(Qt.CursorShape.PointingHandCursor)

    def _toggled(checked: bool) -> None:
        set_folded(group, not checked)
        if on_toggle is not None:
            on_toggle(_title_without_arrow(group.title()), not checked)

    group.toggled.connect(_toggled)
    set_folded(group, folded)
    return group


def apply_to_panel(
    panel: QWidget,
    folded_titles: "set[str] | None" = None,
    on_toggle=None,
    skip: "set[str] | None" = None,
) -> list[QGroupBox]:
    """
    Make every group box in *panel* foldable.

    `folded_titles` names the ones to start folded — normally what the user left
    folded last time. `skip` names groups that should stay fixed, for a panel
    whose single group folding away would leave nothing at all.

    Nested groups are included. A plugin that injects a section into a workflow
    tab wraps its own groups in one of its own, and the Measure tab is ten of
    twelve groups deep — excluding them left the most crowded panel in the
    application with nothing to fold.
    """
    folded_titles = folded_titles or set()
    skip = skip or set()
    done: list[QGroupBox] = []
    groups = panel.findChildren(QGroupBox)
    for group in groups:
        title = _title_without_arrow(group.title())
        if not title or title in skip:
            continue
        make_collapsible(group, folded=title in folded_titles, on_toggle=on_toggle)
        done.append(group)
    return done
