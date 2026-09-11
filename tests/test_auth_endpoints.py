from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from auth.security import hash_password
from auth.users_repository import create_user

from api.main import app

from conftest import FakeFirestoreClient, FakeFirestoreRuntime


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def fake_firestore(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> FakeFirestoreClient:
    fake_client = FakeFirestoreClient()
    monkeypatch.setattr(client.app.state, "firestore", FakeFirestoreRuntime(client=fake_client))
    return fake_client


def test_login_with_valid_credentials_returns_token(client: TestClient, fake_firestore: FakeFirestoreClient) -> None:
    create_user(fake_firestore, "fulano@energisa.com.br", hash_password("senha-correta"))

    response = client.post("/auth/login", json={"email": "fulano@energisa.com.br", "senha": "senha-correta"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]
    assert payload["expires_in"] > 0


def test_login_with_wrong_password_returns_401_generic_message(
    client: TestClient, fake_firestore: FakeFirestoreClient
) -> None:
    create_user(fake_firestore, "fulano@energisa.com.br", hash_password("senha-correta"))

    response = client.post("/auth/login", json={"email": "fulano@energisa.com.br", "senha": "senha-errada"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciais inválidas."


def test_login_with_unknown_user_returns_401_generic_message(
    client: TestClient, fake_firestore: FakeFirestoreClient
) -> None:
    response = client.post("/auth/login", json={"email": "desconhecido@energisa.com.br", "senha": "qualquer"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Credenciais inválidas."


def test_login_with_non_corporate_email_returns_422(client: TestClient, fake_firestore: FakeFirestoreClient) -> None:
    response = client.post("/auth/login", json={"email": "fulano@gmail.com", "senha": "qualquer"})

    assert response.status_code == 422


def test_login_without_firestore_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client.app.state, "firestore", None)

    response = client.post("/auth/login", json={"email": "fulano@energisa.com.br", "senha": "qualquer"})

    assert response.status_code == 503


def test_login_with_inactive_user_returns_401(client: TestClient, fake_firestore: FakeFirestoreClient) -> None:
    user = create_user(fake_firestore, "fulano@energisa.com.br", hash_password("senha-correta"))
    fake_firestore.collection("users").document(user.email).set(
        {"email": user.email, "password_hash": user.password_hash, "criado_em": user.criado_em, "ativo": False}
    )

    response = client.post("/auth/login", json={"email": "fulano@energisa.com.br", "senha": "senha-correta"})

    assert response.status_code == 401
