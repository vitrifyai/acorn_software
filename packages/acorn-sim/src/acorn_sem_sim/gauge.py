"""A live scale gauge: probe, pixel, interaction volume, field -- on one axis.

This is the piece of feedback SEM simulation actually needs and almost nothing
provides. The single most common way to get a meaningless SEM image, real or
simulated, is to pick a beam energy whose interaction volume is comparable to
the field of view: everything then blurs into everything else and no amount of
dose or dwell time recovers it. The relationship is not obvious from the
numbers, because it spans five orders of magnitude -- a 1 nm probe and a 3 um
interaction volume do not sit on the same linear axis.

So it is drawn on a log axis, with the interaction volume as a band rather than
a tick, and colour-coded against the one rule that matters: the interaction
volume should be small compared to the field you are imaging.

The range is the Kanaya-Okayama formula, not a Monte Carlo run, so this updates
instantly as sliders move. The MC refines it; KO is right to within a factor
that does not change the advice.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .materials import get, kanaya_okayama_nm

_LO_NM = 0.1
_HI_NM = 1e5            # 100 um

# Verdict colours, mirroring ACORN's annotation palette conventions.
_GOOD = QColor("#5aa469")
_WARN = QColor("#d4a24c")
_BAD = QColor("#c65f5f")
_INK = QColor("#6c7086")


def _frac(nm: float) -> float:
    """Position of a length on the log axis, clamped to [0, 1]."""
    import math
    nm = max(min(float(nm), _HI_NM), _LO_NM)
    return (math.log10(nm) - math.log10(_LO_NM)) / (math.log10(_HI_NM) - math.log10(_LO_NM))


def _fmt(nm: float) -> str:
    if nm >= 1e6:
        return f"{nm / 1e6:.3g} mm"
    if nm >= 1000:
        return f"{nm / 1000:.3g} um"
    if nm >= 1:
        return f"{nm:.3g} nm"
    return f"{nm * 10:.3g} A"


class InteractionVolumeGauge(QWidget):
    """Draws probe / pixel / interaction volume / field on a shared log axis."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._probe_nm = 1.0
        self._pixel_nm = 4.0
        self._field_nm = 2048.0
        self._range_nm = 500.0
        self._material = "carbon"
        self._kv = 5.0
        self.setToolTip(
            "Interaction volume of the material you are annotating, against "
            "the field of view.\n"
            "Electrons scatter sideways inside the specimen, so a feature's "
            "boundary cannot be located more precisely than its own interaction "
            "volume -- no matter how small the pixel or how long the dwell.\n"
            "Lower the beam energy or pick a heavier material to shrink it.")

    def update_scales(self, material: str, kv: float, pixel_nm: float,
                      field_px: int, probe_nm: float) -> None:
        self._material = material
        self._kv = float(kv)
        self._pixel_nm = float(pixel_nm)
        self._probe_nm = float(probe_nm)
        self._field_nm = float(pixel_nm) * int(field_px)
        try:
            self._range_nm = kanaya_okayama_nm(get(material), float(kv))
        except (KeyError, ZeroDivisionError):
            self._range_nm = 0.0
        self.update()

    # -- the rule this whole widget exists to communicate ---------------------
    def verdict(self) -> tuple[str, QColor]:
        if self._range_nm <= 0:
            return "No interaction volume (vacuum).", _INK
        ratio = self._field_nm / self._range_nm
        if ratio >= 8:
            return ((f"Interaction volume is 1/{ratio:.0f} of the field -- "
                     f"features stay separable."), _GOOD)
        if ratio >= 2:
            return ((f"Interaction volume is 1/{ratio:.1f} of the field. Usable, "
                     f"but fine detail will blur."), _WARN)
        return ((f"Interaction volume is {1 / ratio:.1f}x the field. Everything "
                 f"blurs into everything else -- lower the kV."), _BAD)

    def advice(self) -> str:
        """One concrete suggestion, or empty if the settings are already sound."""
        if self._range_nm <= 0 or self._field_nm / self._range_nm >= 8:
            return ""
        for kv in (20.0, 10.0, 5.0, 2.0, 1.0, 0.5):
            if kv >= self._kv:
                continue
            try:
                r = kanaya_okayama_nm(get(self._material), kv)
            except (KeyError, ZeroDivisionError):
                continue
            if r > 0 and self._field_nm / r >= 8:
                return f"Try {kv:g} kV -- that brings it to {_fmt(r)}."
        return "Even 0.5 kV will not fit this field; widen the field of view."

    def paintEvent(self, _event) -> None:      # Qt naming, not snake_case
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 10
        x0, x1 = pad, w - pad
        label_top = 4              # marker labels, possibly two rows
        bar_top = label_top + 30
        bar_h = 20
        bar_bot = bar_top + bar_h
        decade_y = bar_bot + 3
        summary_y = h - 17

        def px(nm):
            return x0 + _frac(nm) * (x1 - x0)

        faint = QColor(_INK.red(), _INK.green(), _INK.blue(), 55)
        mid = QColor(_INK.red(), _INK.green(), _INK.blue(), 155)
        small = QFont(self.font())
        small.setPointSizeF(max(6.5, self.font().pointSizeF() - 2.5))

        # decade grid
        p.setFont(small)
        for decade_nm, tag in ((1, "1nm"), (10, "10nm"), (100, "100nm"),
                               (1000, "1um"), (10_000, "10um"), (100_000, "100um")):
            gx = px(decade_nm)
            p.setPen(QPen(faint, 1))
            p.drawLine(int(gx), bar_top, int(gx), bar_bot)
            p.setPen(QPen(mid, 1))
            # Clamp inside the widget: the last decade sits on the right edge,
            # so a centred box would hang half off and clip the label.
            lx = min(max(gx - 24, 0.0), float(w - 48))
            p.drawText(QRectF(lx, decade_y, 48, 12),
                       Qt.AlignmentFlag.AlignHCenter, tag)

        # interaction volume, drawn as the band it is
        _, colour = self.verdict()
        if self._range_nm > 0:
            bx0, bx1 = px(self._probe_nm), px(self._range_nm)
            band = QColor(colour)
            band.setAlpha(70)
            p.fillRect(QRectF(bx0, bar_top, max(bx1 - bx0, 2), bar_h), band)
            p.setPen(QPen(colour, 2))
            p.drawLine(int(bx1), bar_top - 3, int(bx1), bar_bot + 2)

        # Marker labels stagger onto a second row when they crowd, which they
        # routinely do -- probe and pixel are often within a nanometre or two of
        # each other and would otherwise overprint into an unreadable smear.
        markers = [(self._probe_nm, "probe", _INK),
                   (self._pixel_nm, "pixel", _INK),
                   (self._field_nm, "field", QColor("#7aa2f7"))]
        markers.sort(key=lambda m: m[0])
        last_x = -1e9
        row = 0
        for nm, tag, col in markers:
            mx = px(nm)
            row = 1 if (mx - last_x) < 44 and row == 0 else 0
            last_x = mx
            p.setPen(QPen(col, 2))
            p.drawLine(int(mx), bar_top - 3, int(mx), bar_bot + 2)
            p.setFont(small)
            lx = min(max(mx - 34, 0.0), float(w - 68))
            p.drawText(QRectF(lx, label_top + row * 13, 68, 13),
                       Qt.AlignmentFlag.AlignHCenter, tag)

        # summary: the number on the left, the verdict on the right
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        p.setPen(QPen(colour, 1))
        head = f"{_fmt(self._range_nm)} in {self._material} at {self._kv:g} kV"
        # Measure with the BOLD font actually used to draw it. Measuring after
        # switching to the small font understates the width and the two strings
        # print on top of each other.
        head_w = QFontMetricsF(f).horizontalAdvance(head) + 16
        p.drawText(QRectF(pad, summary_y, w - 2 * pad, 15),
                   Qt.AlignmentFlag.AlignLeft, head)

        remaining = w - 2 * pad - head_w
        verdict = self.verdict()[0].split(" -- ")[0].rstrip(".")
        p.setFont(small)
        if remaining >= QFontMetricsF(small).horizontalAdvance(verdict):
            p.drawText(QRectF(pad + head_w, summary_y, remaining, 15),
                       Qt.AlignmentFlag.AlignRight, verdict)
        # Too narrow: the band colour already carries the verdict, and the
        # advice line underneath spells it out. Better blank than overprinted.
        p.end()
