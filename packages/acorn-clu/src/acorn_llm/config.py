"""LLM provider configuration — loaded from / saved to ~/.acorn/llm_config.json."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

_CONFIG_PATH = Path.home() / ".acorn" / "llm_config.json"


@dataclass
class LLMConfig:
    provider: str = "anthropic"       # "anthropic" | "openai_compat"
    model: str = "claude-opus-4-7"    # vision / primary model
    tool_model: str = ""              # tool-capable model (Ollama only); empty = same as model
    api_key: str = ""
    base_url: str = ""                # for openai_compat (Ollama, Groq, Together…)
    max_tokens: int = 4096
    include_image: bool = True
    image_max_px: int = 1024


def load_config() -> LLMConfig:
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text())
            return LLMConfig(**{k: v for k, v in data.items() if k in LLMConfig.__dataclass_fields__})
        except Exception:
            pass
    return LLMConfig()


def save_config(cfg: LLMConfig) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))
