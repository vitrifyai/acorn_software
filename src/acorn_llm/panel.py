"""AI Assistant panel — chat UI with streaming, tool status, and confirm dialogs."""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton,
    QSizePolicy, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
    QComboBox,
)

from acorn_llm.config import LLMConfig, load_config, save_config
from acorn_llm.agent import LLMAgent

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext

# ── colour constants (ORNL theme) ─────────────────────────────────────────────
_BG_USER = "#1e3a1e"
_BG_ASST = "#1e2a30"
_BG_TOOL = "#1a2535"
_BG_WARN = "#3a2a1a"
_BG_NOTE = "transparent"
_FG_MAIN = "#e0e0e0"
_FG_DIM  = "#888888"


class AssistantPanel(QWidget):
    def __init__(self, context: "AcornContext", parent=None) -> None:
        super().__init__(parent)
        self._context  = context
        self._config   = load_config()
        self._messages: list[dict] = []    # [{role, content}, …] conversation history
        self._items: list[tuple]   = []    # [(kind, text), …] display items
        self._stream_text = ""             # accumulator for current streaming turn
        self._streaming   = False
        self._agent: Optional[LLMAgent] = None

        # throttle re-render to ≤20 fps while streaming
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(50)
        self._render_timer.timeout.connect(self._render)

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── settings ──────────────────────────────────────────────────
        self._settings_box = QGroupBox("Provider Settings")
        self._settings_box.setCheckable(True)
        self._settings_box.setChecked(False)
        sf = QFormLayout(self._settings_box)
        sf.setSpacing(4)

        self._provider_combo = QComboBox()
        self._provider_combo.addItem("Anthropic (Claude)", "anthropic")
        self._provider_combo.addItem("OpenAI / compatible (GPT, Ollama, Groq…)", "openai_compat")
        i = self._provider_combo.findData(self._config.provider)
        self._provider_combo.setCurrentIndex(max(i, 0))
        self._provider_combo.currentIndexChanged.connect(self._sync_base_url_row)
        sf.addRow("Provider:", self._provider_combo)

        self._model_edit = QLineEdit(self._config.model)
        self._model_edit.setPlaceholderText("Vision model  e.g. acorn-llm / claude-opus-4-7")
        sf.addRow("Vision model:", self._model_edit)

        self._tool_model_edit = QLineEdit(self._config.tool_model)
        self._tool_model_edit.setPlaceholderText("Tool model  e.g. acorn-tools  (leave blank to use same)")
        sf.addRow("Tool model:", self._tool_model_edit)

        self._key_edit = QLineEdit(self._config.api_key)
        self._key_edit.setPlaceholderText("API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)")
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        sf.addRow("API key:", self._key_edit)

        self._url_label = QLabel("Base URL:")
        self._url_edit  = QLineEdit(self._config.base_url)
        self._url_edit.setPlaceholderText("http://localhost:11434/v1  (Ollama default)")
        sf.addRow(self._url_label, self._url_edit)

        save_btn = QPushButton("Save settings")
        save_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        save_btn.clicked.connect(self._save_settings)
        sf.addRow("", save_btn)

        self._sync_base_url_row()
        layout.addWidget(self._settings_box)

        # ── options row ───────────────────────────────────────────────
        opts = QHBoxLayout()
        self._img_chk = QCheckBox("Include image")
        self._img_chk.setChecked(self._config.include_image)
        self._img_chk.setToolTip("Attach a thumbnail of the current image (vision models only).")

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Vision", "vision")
        self._mode_combo.addItem("Tools", "tools")
        self._mode_combo.addItem("Auto", "auto")
        self._mode_combo.setCurrentIndex(2)
        self._mode_combo.setToolTip(
            "Vision: use vision model, no tools\n"
            "Tools: use tool model, tool dispatch enabled\n"
            "Auto: vision model when image included, tool model otherwise"
        )
        self._mode_combo.setFixedWidth(80)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(56)
        clear_btn.clicked.connect(self._clear)
        opts.addWidget(self._img_chk)
        opts.addWidget(self._mode_combo)
        opts.addStretch()
        opts.addWidget(clear_btn)
        layout.addLayout(opts)

        # ── chat display ──────────────────────────────────────────────
        self._chat = QTextBrowser()
        self._chat.setReadOnly(True)
        self._chat.setOpenExternalLinks(False)
        self._chat.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._chat, 1)

        # ── input ─────────────────────────────────────────────────────
        self._input = QTextEdit()
        self._input.setFixedHeight(68)
        self._input.setPlaceholderText("Ask something… (Enter to send, Shift+Enter for newline)")
        self._input.installEventFilter(self)
        layout.addWidget(self._input)

        send_btn = QPushButton("Send")
        send_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        send_btn.clicked.connect(self._send)
        layout.addWidget(send_btn)

    def _sync_base_url_row(self) -> None:
        compat = self._provider_combo.currentData() == "openai_compat"
        self._url_label.setVisible(compat)
        self._url_edit.setVisible(compat)

    def _save_settings(self) -> None:
        self._config.provider      = self._provider_combo.currentData()
        self._config.model         = self._model_edit.text().strip()
        self._config.tool_model    = self._tool_model_edit.text().strip()
        self._config.api_key       = self._key_edit.text().strip()
        self._config.base_url      = self._url_edit.text().strip()
        self._config.include_image = self._img_chk.isChecked()
        save_config(self._config)
        self._settings_box.setChecked(False)
        self._add_note("Settings saved.")

    def _clear(self) -> None:
        self._messages.clear()
        self._items.clear()
        self._stream_text = ""
        self._streaming   = False
        self._render()

    # ------------------------------------------------------------------
    # Key filter — Enter sends, Shift+Enter inserts newline
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            mods = event.modifiers()
            key  = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not (mods & Qt.KeyboardModifier.ShiftModifier):
                    self._send()
                    return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def _send(self) -> None:
        text = self._input.toPlainText().strip()
        print(f"[Panel._send] text={repr(text[:40])} provider={self._config.provider} model={self._config.model}", flush=True)
        if not text:
            return
        if self._agent and self._agent.isRunning():
            self._add_note("Please wait for the current response to finish.")
            return

        # Basic validation
        if not self._config.model:
            self._add_note("Configure a model in Provider Settings first.")
            return
        if self._config.provider == "anthropic" and not self._config.api_key:
            import os
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self._add_note("Set an API key in Provider Settings (or ANTHROPIC_API_KEY env var).")
                return

        self._input.clear()
        self._messages.append({"role": "user", "content": text})
        self._items.append(("user", text))

        state     = self._context.get_llm_state()
        image_b64 = None
        if self._img_chk.isChecked():
            image_b64 = self._context.get_thumbnail(self._config.image_max_px)

        # Route to vision or tool model based on mode
        import dataclasses
        cfg = dataclasses.replace(self._config)
        mode = self._mode_combo.currentData()
        use_tool_model = (
            mode == "tools"
            or (mode == "auto" and image_b64 is None)
        )
        if use_tool_model and cfg.tool_model:
            cfg.model = cfg.tool_model

        self._agent = LLMAgent(cfg, self._messages, state, image_b64)
        self._agent.token_emitted.connect(self._on_token)
        self._agent.tool_called.connect(self._on_tool)
        self._agent.confirm_needed.connect(self._on_confirm)
        self._agent.done.connect(self._on_done)
        self._agent.error.connect(self._on_error)

        self._stream_text = ""
        self._streaming   = True
        self._render_timer.start()
        self._agent.start()

    # ------------------------------------------------------------------
    # Agent slots
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_token(self, text: str) -> None:
        self._stream_text += text

    @pyqtSlot(str, dict)
    def _on_tool(self, name: str, params: dict) -> None:
        self._items.append(("tool", _tool_label(name, params)))
        self._context.action_requested.emit(name, params)
        self._render()

    @pyqtSlot(str, str, dict)
    def _on_confirm(self, name: str, summary: str, params: dict) -> None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirm action")
        dlg.setText(summary)
        dlg.setInformativeText("Proceed?")
        dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
        if dlg.exec() == QMessageBox.StandardButton.Yes:
            self._items.append(("tool", _tool_label(name, params)))
            self._context.action_requested.emit(name, params)
        else:
            self._items.append(("warn", f"Cancelled: {name}"))
        self._render()

    @pyqtSlot()
    def _on_done(self) -> None:
        self._render_timer.stop()
        if self._stream_text.strip():
            self._messages.append({"role": "assistant", "content": self._stream_text})
            self._items.append(("asst", self._stream_text))
        self._stream_text = ""
        self._streaming   = False
        self._render()

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._render_timer.stop()
        self._streaming   = False
        self._stream_text = ""
        self._add_note(f"Error: {msg}")

    def _add_note(self, text: str) -> None:
        self._items.append(("note", text))
        self._render()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        sb      = self._chat.verticalScrollBar()
        at_end  = sb.value() >= sb.maximum() - 30

        parts = [
            '<html><body style="background:#1a1a1a;color:#e0e0e0;'
            'font-family:Ubuntu,\'Noto Sans\',sans-serif;font-size:12px;margin:4px;">'
        ]

        for kind, text in self._items:
            parts.append(_render_item(kind, text))

        if self._streaming:
            cursor_char = "█" if (len(self._stream_text) % 4) < 2 else " "
            parts.append(_render_item("asst_stream", self._stream_text, cursor=cursor_char))

        if not self._items and not self._streaming:
            parts.append(
                '<p style="color:#888888;text-align:center;margin-top:40px;">'
                'Ask a question or give an instruction.<br>'
                'Examples: "find lamella"  &bull;  "segment vesicles"  &bull;  "start training"'
                '</p>'
            )

        parts.append("</body></html>")
        self._chat.setHtml("".join(parts))

        if at_end:
            sb.setValue(sb.maximum())


# ------------------------------------------------------------------
# Render helpers
# ------------------------------------------------------------------

def _render_item(kind: str, text: str, cursor: str = "") -> str:
    e = _esc(text)
    if kind == "user":
        return (
            f'<div style="background:{_BG_USER};border-radius:6px;'
            f'padding:6px 8px;margin:4px 0;">'
            f'<span style="color:#4dbb78;font-weight:bold;">You</span><br>'
            f'{e.replace(chr(10), "<br>")}'
            f'</div>'
        )
    if kind in ("asst", "asst_stream"):
        return (
            f'<div style="background:{_BG_ASST};border-radius:6px;'
            f'padding:6px 8px;margin:4px 0;">'
            f'<span style="color:#4d8ec4;font-weight:bold;">CLU</span><br>'
            f'{e.replace(chr(10), "<br>")}{cursor}'
            f'</div>'
        )
    if kind == "tool":
        return (
            f'<div style="background:{_BG_TOOL};border-radius:4px;'
            f'padding:3px 8px;margin:2px 0;color:#888888;font-size:11px;">'
            f'&#9654; <i>{e}</i>'
            f'</div>'
        )
    if kind == "warn":
        return (
            f'<div style="background:{_BG_WARN};border-radius:4px;'
            f'padding:3px 8px;margin:2px 0;color:#c0392b;font-size:11px;">'
            f'&#9654; <i>{e}</i>'
            f'</div>'
        )
    # note
    return (
        f'<div style="color:{_FG_DIM};font-size:11px;padding:1px 4px;">'
        f'{e}'
        f'</div>'
    )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tool_label(name: str, params: dict) -> str:
    p = params
    m = {
        "run_sam_auto":       f"SAM auto-segment  label={p.get('label','')}  pts/side={p.get('points_per_side', 32)}",
        "run_yolo_detect":    f"YOLO detect  label={p.get('label','')}",
        "run_yolo_segment":   f"YOLO segment  label={p.get('label','')}",
        "accept_annotations": f"Accept {p.get('model','all')} annotations",
        "queue_for_export":   "Queue image for training export",
        "start_training":     f"Start training — {p.get('summary','')}",
        "finalize_dataset":   f"Finalize dataset — {p.get('summary','')}",
    }
    return m.get(name, f"{name}")
