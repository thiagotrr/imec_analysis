"""Endpoint de autenticação (Task 010).

Único endpoint HTTP de auth: ``POST /auth/login``. Não há cadastro público —
usuários são criados via ``scripts/db/firestore/gerenciar_usuarios.py``
(decisão confirmada, ver ``docs/task010_*.md``).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from auth.security import verify_password, create_access_token
from auth.settings import load_jwt_settings
from auth.users_repository import get_user

from log import get_log

from ..models.auth import LoginRequest, TokenResponse

log = get_log()

TAG = "Autenticação"

router = APIRouter()

_RESPONSE_401_LOGIN_INVALID = {
    "description": "E-mail ou senha inválidos (mensagem genérica, não indica qual dos dois está incorreto).",
}
_RESPONSE_503_FIRESTORE = {
    "description": "Firestore indisponível para consultar o usuário.",
}


@router.post(
    "/auth/login",
    tags=[TAG],
    summary="Autentica um usuário e emite um token JWT",
    description=(
        "Recebe e-mail corporativo (domínio `@energisa.com.br`) e senha; retorna um token JWT "
        "(HS256, stateless, sem refresh) a ser enviado em `Authorization: Bearer <token>` nas "
        "rotas protegidas de `/inspecao` e `/historico`. Não há endpoint de cadastro — usuários "
        "são criados via script administrativo."
    ),
    response_model=TokenResponse,
    responses={401: _RESPONSE_401_LOGIN_INVALID, 422: {"description": "Payload inválido."}, 503: _RESPONSE_503_FIRESTORE},
)
def login(credentials: LoginRequest, request: Request) -> TokenResponse:
    firestore_runtime = getattr(request.app.state, "firestore", None)
    if firestore_runtime is None:
        raise HTTPException(status_code=503, detail="Firestore indisponível no momento.")

    try:
        user = get_user(firestore_runtime.client, credentials.email)
    except Exception as exc:  # falha de conectividade/permissão do SDK
        log.exception("Falha ao consultar usuário no Firestore em /auth/login")
        raise HTTPException(status_code=503, detail=f"Firestore indisponível: {exc}") from exc

    if user is None or not user.ativo or not verify_password(credentials.senha, user.password_hash):
        log.info("Tentativa de login rejeitada para %s", credentials.email)
        raise HTTPException(status_code=401, detail="Credenciais inválidas.")

    jwt_settings = getattr(request.app.state, "jwt_settings", None) or load_jwt_settings()
    token = create_access_token(user.email, jwt_settings)
    log.info("Login bem-sucedido para %s", user.email)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=jwt_settings.expire_minutes * 60,
    )
