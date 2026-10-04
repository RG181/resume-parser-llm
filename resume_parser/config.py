"""Configuration and prompt loading. Prompts and settings live in files, not in code."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml (+ .env). Env var RESUME_PARSER_MODEL overrides llm.model."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # python-dotenv is optional
        pass
    path = Path(path or os.getenv("RESUME_PARSER_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if os.getenv("RESUME_PARSER_PROVIDER"):
        cfg["llm"]["provider"] = os.environ["RESUME_PARSER_PROVIDER"]
    if os.getenv("RESUME_PARSER_MODEL"):
        cfg["llm"]["model"] = os.environ["RESUME_PARSER_MODEL"]
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: dict, relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


class Prompts:
    """Thin wrapper around prompts.yaml with {{placeholder}} rendering."""

    def __init__(self, data: dict):
        self.data = data

    @property
    def version(self) -> str:
        return str(self.data.get("version", "?"))

    def render(self, name: str, **values) -> str:
        if name not in self.data:
            raise KeyError(f"Prompt '{name}' not found in prompt file")
        text = self.data[name]
        for key, val in values.items():
            text = text.replace("{{" + key + "}}", str(val))
        return text.strip()


def load_prompts(cfg: dict) -> Prompts:
    with open(resolve_path(cfg, cfg["prompts"]["file"]), encoding="utf-8") as fh:
        return Prompts(yaml.safe_load(fh))
