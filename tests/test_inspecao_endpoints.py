from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.inspecao_request_model import LaudoCompletoRequest, LaudoSinteticoRequest
from api.inspecao_response_model import ModeloInfoResponse
from api.inspecao_services import obter_info_modelos

TAG = "Inspeção de Medidor de Consumo"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _laudo_completo_example() -> dict[str, object]:
    return dict(LaudoCompletoRequest.model_config["json_schema_extra"]["example"])


def _laudo_sintetico_example() -> dict[str, object]:
    return dict(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])


# ---------------------------------------------------------------------------
# POST /inspecao/laudo_completo
# ---------------------------------------------------------------------------


def test_laudo_completo_valid_payload_returns_documented_500_stub(client: TestClient) -> None:
    response = client.post("/inspecao/laudo_completo", json=_laudo_completo_example())

    assert response.status_code == 500
    assert "não implementado" in response.json()["detail"]


def test_laudo_completo_invalid_payload_returns_422(client: TestClient) -> None:
    response = client.post("/inspecao/laudo_completo", json={"campo_que_nao_existe": 1})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /inspecao/laudo_sintetico
# ---------------------------------------------------------------------------


def test_laudo_sintetico_valid_payload_returns_documented_500_stub(client: TestClient) -> None:
    response = client.post("/inspecao/laudo_sintetico", json=_laudo_sintetico_example())

    assert response.status_code == 500
    assert "não implementado" in response.json()["detail"]


def test_laudo_sintetico_invalid_payload_returns_422(client: TestClient) -> None:
    response = client.post("/inspecao/laudo_sintetico", json={})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /inspecao/csv
# ---------------------------------------------------------------------------


def _build_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    return buffer.getvalue().encode("utf-8")


def test_csv_upload_with_valid_rows_returns_documented_500_stub(client: TestClient) -> None:
    csv_bytes = _build_csv_bytes([_laudo_completo_example()])

    response = client.post("/inspecao/csv", files={"arquivo": ("laudos.csv", csv_bytes, "text/csv")})

    assert response.status_code == 500
    assert "não implementado" in response.json()["detail"]


def test_csv_upload_with_invalid_row_returns_422_with_per_row_errors(client: TestClient) -> None:
    csv_bytes = "campo_que_nao_existe\n1\n".encode("utf-8")

    response = client.post("/inspecao/csv", files={"arquivo": ("laudos.csv", csv_bytes, "text/csv")})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "linhas_invalidas" in detail
    assert detail["linhas_invalidas"][0]["numero_linha"] == 1


def test_csv_upload_with_empty_file_returns_422(client: TestClient) -> None:
    response = client.post("/inspecao/csv", files={"arquivo": ("laudos.csv", b"", "text/csv")})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /inspecao/modelos
# ---------------------------------------------------------------------------


def test_listar_modelos_returns_200_even_without_compiled_models(client: TestClient) -> None:
    response = client.get("/inspecao/modelos")

    assert response.status_code == 200
    payload = response.json()
    assert "champion" in payload
    assert "message" in payload


def test_obter_info_modelos_service_reports_champion_when_present(tmp_path: Path) -> None:
    """Testa o service isoladamente (sem depender do estado real de
    `model/compiled/`) simulando um `champion.json` já compilado em um
    diretório temporário."""
    compiled_dir = tmp_path / "compiled"
    compiled_dir.mkdir(parents=True)
    champion_metadata = {
        "algorithm": "xgboost",
        "resampling": "smote",
        "trained_at": "2026-07-24T22:00:00+00:00",
        "feature_columns": ["svd_001", "svd_002"],
        "target_classes": ["1", "10"],
        "metrics": {"f1": 0.6114, "accuracy": 0.9344},
        "tier_thresholds": {"A": 15.0, "B": 1.0, "C": 0.1},
        "class_weight_registry_path": None,
    }
    (compiled_dir / "champion.json").write_text(json.dumps(champion_metadata), encoding="utf-8")

    result = obter_info_modelos(output_dir=tmp_path)

    assert isinstance(result, ModeloInfoResponse)
    assert result.champion is not None
    assert result.champion.algorithm == "xgboost"
    assert result.champion.resampling == "smote"
    assert result.champion.feature_columns_count == 2
    assert result.message is None


# ---------------------------------------------------------------------------
# OpenAPI / Swagger (`/docs`)
# ---------------------------------------------------------------------------


def test_openapi_schema_includes_all_four_endpoints_with_expected_tag(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    expected_operations = {
        ("/inspecao/laudo_completo", "post"),
        ("/inspecao/laudo_sintetico", "post"),
        ("/inspecao/csv", "post"),
        ("/inspecao/modelos", "get"),
    }
    found_operations = set()
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if (path, method) in expected_operations:
                found_operations.add((path, method))
                assert TAG in operation.get("tags", [])
                assert operation.get("summary")
                assert operation.get("description")
                assert "422" in operation.get("responses", {}) or "500" in operation.get("responses", {})

    assert found_operations == expected_operations


def test_docs_endpoint_is_served(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
