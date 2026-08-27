"""Contrast control panel widget."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QSizePolicy,
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from acorn.core.contrast import ContrastParams


# ── preset persistence ────────────────────────────────────────────────────────

_PRESET_FILE = Path.home() / ".acorn" / "presets.json"

_BUILTIN_PRESETS: dict[str, ContrastParams] = {
    "Percentile 0.5/99.5 (standard annotation for low dose)": ContrastParams(
        method="percentile", low_pct=0.5, high_pct=99.5
    ),
    "Default (Bandpass)": ContrastParams(),
    "Bandpass Aggressive": ContrastParams(
        method="bandpass", bp_low_sigma=40.0, bp_high_sigma=0.5
    ),
    "Fourier Bandpass": ContrastParams(method="fourier", fbp_hp_px=100.0, fbp_lp_px=4.0),
    "Percentile 1/99": ContrastParams(
        method="percentile", low_pct=1.0, high_pct=99.0
    ),
    "Sigma 3": ContrastParams(method="sigma", n_sigma=3.0),
    "Sigma 5": ContrastParams(method="sigma", n_sigma=5.0),
    "Adaptive CLAHE": ContrastParams(method="adaptive", clip_limit=0.03),
}


def _load_user_presets() -> dict[str, dict]:
    if _PRESET_FILE.exists():
        try:
            return json.loads(_PRESET_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_user_presets(presets: dict[str, dict]) -> None:
    _PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PRESET_FILE.write_text(json.dumps(presets, indent=2))


def _params_to_dict(p: ContrastParams) -> dict:
    return asdict(p)


def _dict_to_params(d: dict) -> ContrastParams:
    return ContrastParams(**{k: v for k, v in d.items() if k in ContrastParams.__dataclass_fields__})


# ── helper widget ─────────────────────────────────────────────────────────────

class _ParamRow(QWidget):
    """Label + slider + spinbox in one row."""

    def __init__(
        self,
        label: str,
        vmin: float,
        vmax: float,
        value: float,
        decimals: int = 1,
        step: float = 0.1,
        parent=None,
    ):
        super().__init__(parent)
        self._scale = 10 ** decimals
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setMinimumWidth(60)
        lbl.setMaximumWidth(100)
        layout.addWidget(lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(vmin * self._scale), int(vmax * self._scale))
        self.slider.setValue(int(value * self._scale))
        layout.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)
        self.spin.setFixedWidth(72)
        layout.addWidget(self.spin)

        # sync
        self.slider.valueChanged.connect(
            lambda v: self.spin.setValue(v / self._scale)
        )
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(v * self._scale))
        )

    @property
    def value(self) -> float:
        return self.spin.value()

    @value.setter
    def value(self, v: float) -> None:
        self.spin.setValue(v)

    def on_change(self, cb) -> None:
        self.spin.valueChanged.connect(lambda _: cb())


# ── main panel ────────────────────────────────────────────────────────────────

class ContrastPanel(QWidget):
    """
    Contrast control panel.

    Emits ``contrast_changed(ContrastParams)`` whenever any control changes.
    """

    contrast_changed = pyqtSignal(object)   # ContrastParams
    bin_factor_changed = pyqtSignal(int)     # analysis binning, needs a reload

    def _on_bin_changed(self, _index: int) -> None:
        factor = int(self._bin_combo.currentData() or 1)
        from acorn.core.binning import describe
        self._bin_note.setText(describe(factor))
        if not self._updating:
            self.bin_factor_changed.emit(factor)

    def bin_factor(self) -> int:
        return int(self._bin_combo.currentData() or 1)

    def set_bin_factor(self, factor: int) -> None:
        """Set without emitting -- for restoring saved state."""
        i = self._bin_combo.findData(int(factor))
        if i < 0:
            return
        was, self._updating = self._updating, True
        try:
            self._bin_combo.setCurrentIndex(i)
        finally:
            self._updating = was
        from acorn.core.binning import describe
        self._bin_note.setText(describe(int(factor)))

    def show_binning_result(self, factor: int, pixel_size_nm: float,
                            cropped_px: tuple = (0, 0)) -> None:
        """Report what the load actually produced, not what was requested.

        The two differ when the image does not divide evenly: rows or columns
        are dropped so the shape divides, and a silently smaller field would be
        a measurement error waiting to happen.
        """
        from acorn.core.binning import describe
        text = describe(int(factor), pixel_size_nm)
        if any(cropped_px):
            text += (f" {cropped_px[0]} row(s) and {cropped_px[1]} column(s) "
                     f"were dropped so the shape divided evenly.")
        self._bin_note.setText(text)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        _content = QWidget()
        layout = QVBoxLayout(_content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── analysis binning ──────────────────────────────────────────────────
        # Deliberately first, above every display control, and worded so it
        # cannot be mistaken for one. Everything else in this panel changes how
        # the image is drawn; this changes the pixels that detection and
        # measurement actually receive, and rescales the pixel size with them.
        bin_box = QGroupBox("Analysis binning")
        bin_layout = QVBoxLayout(bin_box)
        bin_layout.setSpacing(4)

        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel("Bin:"))
        self._bin_combo = QComboBox()
        for f in (1, 2, 4, 8):
            self._bin_combo.addItem("None" if f == 1 else f"{f} x {f}", f)
        self._bin_combo.setToolTip(
            "Averages blocks of pixels BEFORE analysis, and scales the pixel "
            "size to match so measurements stay in real units.\n\n"
            "This is not a display setting — it changes the pixels that "
            "detection and measurement receive.\n\n"
            "It does not make faint objects detectable: a filter matched to the "
            "object size extracts the same signal from full resolution. What it "
            "does is reduce a large image properly, by averaging, rather than "
            "leaving a model to resize it however it happens to.\n\n"
            "Applies when an image is loaded, so changing it reloads the "
            "current image.")
        bin_row.addWidget(self._bin_combo, 1)
        bin_layout.addLayout(bin_row)

        self._bin_note = QLabel("")
        self._bin_note.setWordWrap(True)
        self._bin_note.setStyleSheet("font-size: 11px; color: #6c7086;")
        bin_layout.addWidget(self._bin_note)

        self._bin_combo.currentIndexChanged.connect(self._on_bin_changed)
        layout.addWidget(bin_box)

        # ── presets ───────────────────────────────────────────────────────────
        preset_box = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_box)
        preset_layout.setSpacing(4)

        self._preset_combo = QComboBox()
        self._preset_combo.setPlaceholderText("-- select a preset --")
        preset_layout.addWidget(self._preset_combo)

        preset_btn_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.setToolTip("Apply selected preset to current controls")
        save_btn = QPushButton("Save Current")
        save_btn.setToolTip("Save current settings as a named preset")
        del_btn  = QPushButton("Delete")
        del_btn.setToolTip("Delete selected preset (built-ins cannot be deleted)")
        preset_btn_row.addWidget(load_btn)
        preset_btn_row.addWidget(save_btn)
        preset_btn_row.addWidget(del_btn)
        preset_layout.addLayout(preset_btn_row)

        layout.addWidget(preset_box)

        # ── method selector ───────────────────────────────────────────────────
        method_box = QGroupBox("Method")
        method_layout = QHBoxLayout(method_box)
        method_layout.addWidget(QLabel("Normalisation:"))
        self._method_combo = QComboBox()
        for key, label in [
            ("fourier",    "Fourier Bandpass (CryoEM standard)"),
            ("bandpass",   "Spatial Bandpass"),
            ("percentile", "Percentile"),
            ("sigma",      "Sigma"),
            ("adaptive",   "Adaptive CLAHE"),
        ]:
            self._method_combo.addItem(label, userData=key)
        method_layout.addWidget(self._method_combo, 1)
        layout.addWidget(method_box)

        # ── per-method stacked panels ─────────────────────────────────────────
        self._stack = QStackedWidget()
        self._pages: dict[str, QWidget] = {}
        self._pages["fourier"]    = self._make_fourier_page()
        self._pages["bandpass"]   = self._make_bandpass_page()
        self._pages["percentile"] = self._make_percentile_page()
        self._pages["sigma"]      = self._make_sigma_page()
        self._pages["adaptive"]   = self._make_adaptive_page()
        for page in self._pages.values():
            # Pack each page's rows to the top. Without this the stack takes the
            # height of its tallest page and the shorter ones spread their two or
            # three rows across it, leaving big gaps that look like missing controls.
            lay = page.layout()
            if lay is not None:
                lay.addStretch()
            self._stack.addWidget(page)
        self._stack.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        idx = self._method_combo.findData("percentile")
        if idx >= 0:
            self._method_combo.blockSignals(True)
            self._method_combo.setCurrentIndex(idx)
            self._method_combo.blockSignals(False)
        self._stack.setCurrentWidget(self._pages["percentile"])
        layout.addWidget(self._stack)

        # ── gamma ─────────────────────────────────────────────────────────────
        # ── denoising ─────────────────────────────────────────────────────────
        # Here rather than in a workspace of its own: denoising is a transform on
        # the raw data, like contrast, and it reaches the same places — the
        # display, whatever a detector is given, and the tiles written for
        # training. Keeping the two together means one record of what was done.
        from acorn.core import denoise as _denoise

        dn_box = QGroupBox("Denoising")
        dn_vb = QVBoxLayout(dn_box)
        dn_vb.setSpacing(4)

        dn_form = QFormLayout()
        self._denoise_method = QComboBox()
        for m in _denoise.METHODS:
            available = _denoise.is_available(m.key)
            label = m.label if available else f"{m.label}  (not installed)"
            self._denoise_method.addItem(label, m.key)
            idx = self._denoise_method.count() - 1
            self._denoise_method.setItemData(idx, m.note, Qt.ItemDataRole.ToolTipRole)
            if not available:
                self._denoise_method.model().item(idx).setEnabled(False)
        self._denoise_method.currentIndexChanged.connect(self._on_denoise_method_changed)
        dn_form.addRow("Method:", self._denoise_method)
        dn_vb.addLayout(dn_form)

        self._denoise_strength = _ParamRow("Strength", 0.0, 1.0, 0.5, decimals=2, step=0.05)
        dn_vb.addWidget(self._denoise_strength)

        self._denoise_res = _ParamRow("Resolution A", 3.0, 100.0, 20.0, decimals=1, step=1.0)
        self._denoise_res.setToolTip(
            "Cut everything finer than this. Needs the image's pixel size; the "
            "status bar shows whether that came from the file header."
        )
        dn_vb.addWidget(self._denoise_res)

        self._denoise_note = QLabel("")
        self._denoise_note.setWordWrap(True)
        self._denoise_note.setStyleSheet("font-size: 11px; color: #8a8a8a;")
        dn_vb.addWidget(self._denoise_note)
        layout.addWidget(dn_box)

        gamma_box = QGroupBox("Post-processing")
        gamma_vb = QVBoxLayout(gamma_box)
        gamma_vb.setSpacing(4)

        gamma_row = QHBoxLayout()
        self._gamma = _ParamRow("Gamma", 0.25, 2.5, 1.0, decimals=2, step=0.05)
        gamma_reset = QPushButton("Reset")
        gamma_reset.setFixedWidth(64)
        gamma_reset.clicked.connect(lambda: setattr(self._gamma, "value", 1.0))
        gamma_row.addWidget(self._gamma)
        gamma_row.addWidget(gamma_reset)
        gamma_vb.addLayout(gamma_row)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap"))
        self._cmap = QComboBox()
        for cm in ["gray", "gray_r", "viridis", "inferno", "plasma", "magma", "hot", "bone"]:
            self._cmap.addItem(cm)
        cmap_row.addWidget(self._cmap, 1)
        gamma_vb.addLayout(cmap_row)

        layout.addWidget(gamma_box)
        layout.addStretch()

        # ── connect signals ───────────────────────────────────────────────────
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        for page in self._pages.values():
            for w in page.findChildren(QDoubleSpinBox):
                w.valueChanged.connect(self._emit)
        self._gamma.on_change(self._emit)
        self._cmap.currentTextChanged.connect(self._emit)

        load_btn.clicked.connect(self._load_preset)
        save_btn.clicked.connect(self._save_preset)
        del_btn.clicked.connect(self._delete_preset)

        self._refresh_preset_combo()
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setWidget(_content)
        _outer = QVBoxLayout(self)
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(_scroll)

    # ── page builders ─────────────────────────────────────────────────────────

    def _make_fourier_page(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(4)
        self._fbp_hp = _ParamRow("BG scale (px)", 0, 500, 100.0, decimals=0, step=10)
        self._fbp_lp = _ParamRow("Noise scale (px)", 0, 20, 4.0, decimals=1, step=0.5)
        info = QLabel(
            "Fourier-space Gaussian bandpass — standard for cryo-EM (RELION/cryoSPARC).\n"
            "BG scale: removes ice/support gradients at scales larger than this (px).\n"
            "Noise scale: suppresses shot noise at scales smaller than this (px)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: palette(mid); font-size: 11px;")
        vb.addWidget(info)
        vb.addWidget(self._fbp_hp)
        vb.addWidget(self._fbp_lp)
        return w

    def _make_bandpass_page(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(4)
        self._bp_low  = _ParamRow("BG radius (px)", 0, 500, 100.0, decimals=0, step=10)
        self._bp_high = _ParamRow("Smooth (px)", 0, 10, 1.0, decimals=2, step=0.25)
        info = QLabel("Removes ice gradient and suppresses shot noise.")
        info.setWordWrap(True)
        info.setStyleSheet("color: palette(mid); font-size: 11px;")
        vb.addWidget(info)
        vb.addWidget(self._bp_low)
        vb.addWidget(self._bp_high)
        return w

    def _make_percentile_page(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(4)
        self._pct_low  = _ParamRow("Low %",  0,  10,  0.5, decimals=1, step=0.1)
        self._pct_high = _ParamRow("High %", 90, 100, 99.5, decimals=1, step=0.1)
        vb.addWidget(self._pct_low)
        vb.addWidget(self._pct_high)
        return w

    def _make_sigma_page(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        self._n_sigma = _ParamRow("Sigma (s)", 0.5, 8, 3.0, decimals=1, step=0.1)
        vb.addWidget(self._n_sigma)
        return w

    def _make_adaptive_page(self) -> QWidget:
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 0, 0, 0)
        self._clahe = _ParamRow("CLAHE clip", 0.005, 0.1, 0.03, decimals=3, step=0.005)
        vb.addWidget(self._clahe)
        return w

    # ── preset helpers ────────────────────────────────────────────────────────

    def _all_presets(self) -> dict[str, ContrastParams]:
        result = dict(_BUILTIN_PRESETS)
        for name, d in _load_user_presets().items():
            try:
                result[name] = _dict_to_params(d)
            except Exception:
                pass
        return result

    def _refresh_preset_combo(self, select: str | None = None) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for name in self._all_presets():
            self._preset_combo.addItem(name)
        if select is not None:
            idx = self._preset_combo.findText(select)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
        else:
            self._preset_combo.setCurrentIndex(-1)
        self._preset_combo.blockSignals(False)

    def _load_preset(self) -> None:
        name = self._preset_combo.currentText()
        if not name:
            return
        presets = self._all_presets()
        if name not in presets:
            return
        self.set_params(presets[name])
        self._emit()

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:", text=""
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in _BUILTIN_PRESETS:
            QMessageBox.warning(
                self, "Cannot Overwrite Built-in",
                f'"{name}" is a built-in preset and cannot be overwritten.\nChoose a different name.'
            )
            return
        user = _load_user_presets()
        if name in user:
            resp = QMessageBox.question(
                self, "Overwrite preset?",
                f'A preset named "{name}" already exists. Overwrite it?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        user[name] = _params_to_dict(self.params())
        try:
            _save_user_presets(user)
        except Exception as e:
            QMessageBox.warning(
                self, "Could not save preset",
                f"Failed to write the presets file:\n{e}",
            )
            return
        self._refresh_preset_combo(select=name)

    def _delete_preset(self) -> None:
        name = self._preset_combo.currentText()
        if not name:
            return
        if name in _BUILTIN_PRESETS:
            QMessageBox.information(
                self, "Built-in Preset",
                f'"{name}" is a built-in preset and cannot be deleted.'
            )
            return
        user = _load_user_presets()
        if name not in user:
            return
        reply = QMessageBox.question(
            self, "Delete Preset",
            f'Delete preset "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            del user[name]
            _save_user_presets(user)
            self._refresh_preset_combo()

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_method_changed(self) -> None:
        method = self.current_method()
        self._stack.setCurrentWidget(self._pages[method])
        self._emit()

    def _emit(self, *_) -> None:
        if not self._updating:
            self.contrast_changed.emit(self.params())

    # ── public API ────────────────────────────────────────────────────────────

    def current_method(self) -> str:
        return self._method_combo.currentData() or "bandpass"

    def set_params(self, p: ContrastParams) -> None:
        """Update all controls to match p without emitting contrast_changed."""
        self._updating = True
        try:
            idx = self._method_combo.findData(p.method)
            if idx >= 0:
                self._method_combo.setCurrentIndex(idx)
            self._stack.setCurrentWidget(self._pages.get(p.method, self._pages["fourier"]))
            self._fbp_hp.value    = p.fbp_hp_px
            self._fbp_lp.value    = p.fbp_lp_px
            self._pct_low.value   = p.low_pct
            self._pct_high.value  = p.high_pct
            self._n_sigma.value   = p.n_sigma
            self._clahe.value     = p.clip_limit
            self._bp_low.value    = p.bp_low_sigma
            self._bp_high.value   = p.bp_high_sigma
            self._gamma.value     = p.gamma
            self._cmap.setCurrentText(p.colormap)
        finally:
            self._updating = False

    def _on_denoise_method_changed(self) -> None:
        """Show only the control the chosen method actually uses."""
        from acorn.core import denoise as _denoise

        key = self._denoise_method.currentData()
        method = next((m for m in _denoise.METHODS if m.key == key), None)
        uses_res = bool(method and method.uses_resolution)
        active = key not in ("none", "", None)
        self._denoise_strength.setVisible(active and not uses_res)
        self._denoise_res.setVisible(active and uses_res)
        note = ""
        if method is not None and active:
            note = method.note
            if method.slow:
                note += "  Slow on a large image — try it on a work region first."
        self._denoise_note.setText(note)
        self.contrast_changed.emit(self.params())

    def params(self) -> ContrastParams:
        return ContrastParams(
            method=self.current_method(),
            fbp_hp_px=self._fbp_hp.value,
            fbp_lp_px=self._fbp_lp.value,
            low_pct=self._pct_low.value,
            high_pct=self._pct_high.value,
            n_sigma=self._n_sigma.value,
            clip_limit=self._clahe.value,
            bp_low_sigma=self._bp_low.value,
            bp_high_sigma=self._bp_high.value,
            gamma=self._gamma.value,
            colormap=self._cmap.currentText(),
            denoise_method=self._denoise_method.currentData() or "none",
            denoise_strength=self._denoise_strength.value,
            denoise_resolution_a=self._denoise_res.value,
        )
