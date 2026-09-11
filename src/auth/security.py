"""Hash de senha e emissão/validação de JWT (Task 010).

Modelo stateless: o token não é armazenado em nenhum lugar do servidor (nem
Firestore, nem memória, nem cache). ``create_access_token`` apenas assina um
payload; ``decode_access_token`` apenas reverifica assinatura + expiração a
cada chamada — não há consulta a banco na validação do dia a dia (só no
login, para buscar o usuário).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from .settings import JwtSettings


class AuthError(Exception):
    """Falha de autenticação: credenciais inválidas ou token ausente/expirado/inválido."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


@dataclass(frozen=True)
class TokenPayload:
    sub: str
    iat: datetime
    exp: datetime
    iss: str


def create_access_token(subject: str, settings: JwtSettings) -> str:
    if not settings.is_configured():
        raise AuthError("JWT_SECRET não configurado.")
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=settings.expire_minutes)
    payload = {"sub": subject, "iat": now, "exp": expires_at, "iss": settings.issuer}
    return jwt.encode(payload, settings.secret, algorithm=settings.algorithm)


def decode_access_token(token: str, settings: JwtSettings) -> TokenPayload:
    if not settings.is_configured():
        raise AuthError("JWT_SECRET não configurado.")
    try:
        payload = jwt.decode(
            token,
            settings.secret,
            algorithms=[settings.algorithm],  # nunca aceitar o `alg` vindo do token
            issuer=settings.issuer,
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"Token inválido ou expirado: {exc}") from exc
    return TokenPayload(
        sub=str(payload["sub"]),
        iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        iss=str(payload.get("iss", "")),
    )
