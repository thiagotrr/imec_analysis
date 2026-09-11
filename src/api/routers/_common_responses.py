"""Constantes de resposta OpenAPI reaproveitadas entre routers (Task 010)."""
from __future__ import annotations

RESPONSE_401_UNAUTHORIZED = {
    "description": (
        "Token ausente, inválido ou expirado — enviar `Authorization: Bearer <token>` "
        "obtido em `POST /auth/login`."
    ),
}
