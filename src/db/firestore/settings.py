"""Configuração de acesso ao Google Firestore via variáveis de ambiente (Task 010)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_DATABASE_ID = "imec-analysis"

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    """Carrega ``.env`` da raiz do projeto (se existir) uma única vez."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)
    _DOTENV_LOADED = True


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class FirestoreSettings:
    """Parâmetros de conexão ao Firestore (client oficial, ADC)."""

    enabled: bool = True
    project_id: str | None = None
    database_id: str = DEFAULT_DATABASE_ID


def load_firestore_settings() -> FirestoreSettings:
    _ensure_dotenv_loaded()
    return FirestoreSettings(
        enabled=_env_bool("FIRESTORE_ENABLED", True),
        project_id=(os.getenv("FIRESTORE_PROJECT_ID") or "").strip() or None,
        database_id=(os.getenv("FIRESTORE_DATABASE_ID") or DEFAULT_DATABASE_ID).strip(),
    )
