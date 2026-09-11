from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from auth.security import create_access_token
from auth.settings import load_jwt_settings

from api.main import app
from api.services.historico import persistir_inferencia
from api.models.inspecao_response import InspecaoLaudoResponse

from conftest import FakeFirestoreClient, FakeFirestoreRuntime


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    token = create_access_token("teste@energisa.com.br", load_jwt_settings())
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def fake_firestore(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> FakeFirestoreClient:
    fake_client = FakeFirestoreClient()
    monkeypatch.setattr(client.app.state, "firestore", FakeFirestoreRuntime(client=fake_client))
    return fake_client


def _response(numero_laudo: str) -> InspecaoLaudoResponse:
    return InspecaoLaudoResponse(
        numero_laudo=numero_laudo,
        classe_prevista="10",
        camada="A",
        resultado="Classe 10 (camada A)",
        resultado_detalhado="Texto detalhado.",
    )


def test_consultar_historico_returns_persisted_entries(
    client: TestClient, fake_firestore: FakeFirestoreClient, auth_headers: dict[str, str]
) -> None:
    persistir_inferencia(fake_firestore, _response("2025006988"), usuario_id="fulano@energisa.com.br")
    persistir_inferencia(fake_firestore, _response("outro-laudo"), usuario_id="fulano@energisa.com.br")

    response = client.get("/historico/2025006988", headers=auth_headers)

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["numero_laudo"] == "2025006988"
    assert items[0]["usuario_id"] == "fulano@energisa.com.br"


def test_consultar_historico_sem_inferencias_retorna_lista_vazia(
    client: TestClient, fake_firestore: FakeFirestoreClient, auth_headers: dict[str, str]
) -> None:
    response = client.get("/historico/laudo-sem-inferencias", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_consultar_historico_without_token_returns_401(
    client: TestClient, fake_firestore: FakeFirestoreClient
) -> None:
    response = client.get("/historico/2025006988")

    assert response.status_code == 401


def test_consultar_historico_without_firestore_returns_503(
    client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client.app.state, "firestore", None)

    response = client.get("/historico/2025006988", headers=auth_headers)

    assert response.status_code == 503


def test_openapi_schema_includes_historico_tag_and_401(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    operation = schema["paths"]["/historico/{numero_laudo}"]["get"]
    assert "Histórico de Inferências" in operation.get("tags", [])
    assert "401" in operation.get("responses", {})
