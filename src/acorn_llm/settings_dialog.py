from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from acorn_llm.config import LLMConfig, load_config, save_config


class LLMSettingsDialog(QDialog):
    """Standalone provider settings — shown when no API key is configured."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Assistant — Configure Provider")
        self.setMinimumWidth(420)

        self._config = load_config()
        layout = QVBoxLayout(self)

        note = QLabel(
            "No API key or base URL is configured.\n"
            "Enter your credentials below, then click Save."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setSpacing(6)

        self._provider = QComboBox()
        self._provider.addItem("Anthropic (Claude)", "anthropic")
        self._provider.addItem("OpenAI / compatible (GPT, Ollama, Groq…)", "openai_compat")
        i = self._provider.findData(self._config.provider)
        self._provider.setCurrentIndex(max(i, 0))
        self._provider.currentIndexChanged.connect(self._sync_url_row)
        form.addRow("Provider:", self._provider)

        self._model = QLineEdit(self._config.model)
        self._model.setPlaceholderText("e.g. claude-opus-4-7")
        form.addRow("Model:", self._model)

        self._key = QLineEdit(self._config.api_key)
        self._key.setPlaceholderText("API key")
        self._key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API key:", self._key)

        self._url_label = QLabel("Base URL:")
        self._url = QLineEdit(self._config.base_url)
        self._url.setPlaceholderText("http://localhost:11434/v1  (Ollama default)")
        form.addRow(self._url_label, self._url)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._sync_url_row()

    def _sync_url_row(self) -> None:
        is_compat = self._provider.currentData() == "openai_compat"
        self._url_label.setVisible(is_compat)
        self._url.setVisible(is_compat)

    def _save(self) -> None:
        cfg = LLMConfig(
            provider=self._provider.currentData(),
            model=self._model.text().strip(),
            api_key=self._key.text().strip(),
            base_url=self._url.text().strip(),
            tool_model=self._config.tool_model,
            max_tokens=self._config.max_tokens,
            include_image=self._config.include_image,
            image_max_px=self._config.image_max_px,
        )
        save_config(cfg)
        self.accept()
