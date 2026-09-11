"""Primeira ``Depends()`` do projeto: valida o JWT em rotas protegidas (Task 010).

Convive com o padrão atual de acesso a ``app.state`` via ``Request`` — não
migra nenhuma rota existente que não precise de auth. Não há consulta ao
Firestore aqui: o token é auto-suficiente (stateless), só a assinatura e o
``exp`` são reverificados a cada chamada (ver ``docs/task010_*.md``).
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.security import AuthError, decode_access_token
from auth.settings import load_jwt_settings

from ..models.auth import AuthenticatedUser

# auto_error=False: por padrão o HTTPBearer do FastAPI levanta 403 quando o
# header `Authorization` está ausente — para manter um único código de erro
# (401) em toda ausência/invalidez/expiração de token, desligamos esse
# comportamento e tratamos a ausência explicitamente abaixo.
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Token ausente.")
    settings = getattr(request.app.state, "jwt_settings", None) or load_jwt_settings()
    try:
        payload = decode_access_token(credentials.credentials, settings)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthenticatedUser(email=payload.sub)
