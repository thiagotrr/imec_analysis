"""Configuração de autenticação JWT via variáveis de ambiente (Task 010)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

# Domínio corporativo aceito para login (decisão fechada — Task 010).
ALLOWED_EMAIL_DOMAIN = "@energisa.com.br"

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv(find_dotenv(usecwd=True), override=False)
    _DOTENV_LOADED = True


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class JwtSettings:
    """Parâmetros de emissão/validação de token JWT (HS256, stateless)."""

    secret: str | None = None
    # Algoritmo fixo em código (não lido de env) — evita downgrade attack.
    algorithm: str = "HS256"
    expire_minutes: int = 480
    issuer: str = "imec-analysis-api"

    def is_configured(self) -> bool:
        return bool(self.secret)


def load_jwt_settings() -> JwtSettings:
    _ensure_dotenv_loaded()
    return JwtSettings(
        secret=(os.getenv("JWT_SECRET") or "").strip() or None,
        expire_minutes=_env_int("JWT_EXPIRE_MINUTES", 480),
    )
