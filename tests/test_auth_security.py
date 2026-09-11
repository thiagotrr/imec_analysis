from __future__ import annotations

import pytest

from auth.security import AuthError, create_access_token, decode_access_token, hash_password, verify_password
from auth.settings import JwtSettings


def test_hash_password_round_trip() -> None:
    hashed = hash_password("minha-senha-123")

    assert hashed != "minha-senha-123"
    assert verify_password("minha-senha-123", hashed)
    assert not verify_password("senha-errada", hashed)


def test_create_and_decode_access_token_round_trip() -> None:
    settings = JwtSettings(secret="segredo-de-teste", expire_minutes=5)

    token = create_access_token("fulano@energisa.com.br", settings)
    payload = decode_access_token(token, settings)

    assert payload.sub == "fulano@energisa.com.br"
    assert payload.iss == settings.issuer
    assert payload.exp > payload.iat


def test_decode_access_token_rejects_wrong_secret() -> None:
    issuing_settings = JwtSettings(secret="segredo-a", expire_minutes=5)
    verifying_settings = JwtSettings(secret="segredo-b", expire_minutes=5)

    token = create_access_token("fulano@energisa.com.br", issuing_settings)

    with pytest.raises(AuthError):
        decode_access_token(token, verifying_settings)


def test_decode_access_token_rejects_expired_token() -> None:
    settings = JwtSettings(secret="segredo-de-teste", expire_minutes=-1)

    token = create_access_token("fulano@energisa.com.br", settings)

    with pytest.raises(AuthError):
        decode_access_token(token, settings)


def test_create_access_token_requires_configured_secret() -> None:
    settings = JwtSettings(secret=None)

    with pytest.raises(AuthError):
        create_access_token("fulano@energisa.com.br", settings)


def test_decode_access_token_algorithm_is_always_fixed_to_hs256() -> None:
    settings = JwtSettings(secret="segredo-de-teste")

    assert settings.algorithm == "HS256"
