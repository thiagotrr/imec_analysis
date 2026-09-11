"""Repositório de usuários de autenticação, coleção Firestore ``users`` (Task 010).

Doc ID = e-mail normalizado (lowercase) — garante unicidade sem query extra.
Funções puras que recebem o client Firestore como parâmetro (mesmo padrão de
``src/api/services/inspecao.py`` recebendo ``runtime``), não o buscam sozinhas.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

COLLECTION = "users"


@dataclass(frozen=True)
class UserRecord:
    email: str
    password_hash: str
    criado_em: datetime
    ativo: bool = True


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user(client: Any, email: str) -> UserRecord | None:
    doc = client.collection(COLLECTION).document(normalize_email(email)).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return UserRecord(
        email=str(data.get("email", doc.id)),
        password_hash=str(data.get("password_hash", "")),
        criado_em=data.get("criado_em") or datetime.now(timezone.utc),
        ativo=bool(data.get("ativo", True)),
    )


def create_user(client: Any, email: str, password_hash: str) -> UserRecord:
    normalized = normalize_email(email)
    criado_em = datetime.now(timezone.utc)
    client.collection(COLLECTION).document(normalized).set(
        {
            "email": normalized,
            "password_hash": password_hash,
            "criado_em": criado_em,
            "ativo": True,
        }
    )
    return UserRecord(email=normalized, password_hash=password_hash, criado_em=criado_em, ativo=True)
