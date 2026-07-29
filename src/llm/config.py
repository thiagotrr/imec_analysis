"""Configuração de provedores LLM via variáveis de ambiente (Task 008)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import find_dotenv, load_dotenv

LlmProvider = Literal["openai", "gemini"]

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_PROVIDER: LlmProvider = "openai"

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    """Carrega ``.env`` da raiz do projeto (se existir) uma única vez."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    # Não sobrescreve variáveis já definidas no ambiente do processo.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    _DOTENV_LOADED = True


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LlmSettings:
    """Parâmetros de integração LLM (LangChain + gate condicional)."""

    enabled: bool = True
    provider: LlmProvider = DEFAULT_PROVIDER
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    # Gate: por default NÃO dispara LLM para classes majoritárias (tier A).
    skip_tier_a: bool = True
    min_proba: float = 0.70
    min_top_gap: float = 0.15
    temperature: float = 0.2
    timeout_seconds: float = 30.0

    @property
    def active_api_key(self) -> str | None:
        if self.provider == "openai":
            return self.openai_api_key
        return self.gemini_api_key

    @property
    def active_model(self) -> str:
        if self.provider == "openai":
            return self.openai_model
        return self.gemini_model

    def is_configured(self) -> bool:
        return bool(self.enabled and self.active_api_key)


def load_llm_settings() -> LlmSettings:
    _ensure_dotenv_loaded()
    provider_raw = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    provider: LlmProvider = "gemini" if provider_raw == "gemini" else "openai"
    return LlmSettings(
        enabled=_env_bool("LLM_ENABLED", True),
        provider=provider,
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        gemini_api_key=(os.getenv("GEMINI_API_KEY") or "").strip() or None,
        openai_model=(os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip(),
        gemini_model=(os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip(),
        skip_tier_a=_env_bool("LLM_SKIP_TIER_A", True),
        min_proba=_env_float("LLM_MIN_PROBA", 0.70),
        min_top_gap=_env_float("LLM_MIN_TOP_GAP", 0.15),
        temperature=_env_float("LLM_TEMPERATURE", 0.2),
        timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 30.0),
    )
