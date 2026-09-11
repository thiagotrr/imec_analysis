"""Contratos Pydantic de autenticação (Task 010)."""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth.settings import ALLOWED_EMAIL_DOMAIN

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginRequest(BaseModel):
    """Corpo de ``POST /auth/login``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"email": "fulano@energisa.com.br", "senha": "********"}},
    )

    email: str = Field(..., description=f"E-mail corporativo (domínio obrigatório: {ALLOWED_EMAIL_DOMAIN}).")
    senha: str = Field(..., description="Senha do usuário.")

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_RE.match(normalized):
            raise ValueError("E-mail em formato inválido.")
        if not normalized.endswith(ALLOWED_EMAIL_DOMAIN):
            raise ValueError(f"Login permitido apenas para e-mails do domínio {ALLOWED_EMAIL_DOMAIN}.")
        return normalized


class TokenResponse(BaseModel):
    """Resposta de ``POST /auth/login``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 28800,
            }
        }
    )

    access_token: str = Field(..., description="Token JWT (HS256) a enviar em `Authorization: Bearer <token>`.")
    token_type: str = Field(default="bearer", description="Sempre `bearer`.")
    expires_in: int = Field(..., description="Tempo de expiração do token, em segundos, a partir da emissão.")


class AuthenticatedUser(BaseModel):
    """Identidade resolvida a partir de um token JWT válido (não persistida por request)."""

    email: str = Field(..., description="E-mail do usuário autenticado (claim `sub` do token).")
