"""Testes unitários da camada de services / narrativa / runtime (Task 007)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from api.inspecao_request_model import LaudoSinteticoRequest
from api.inspecao_services import analisar_laudo_sintetico, analisar_csv_upload
from api.model_runtime import ClassTierInfo, ModelRuntime, ModelRuntimeError, load_model_runtime
from api.narrative import compose_resultado, compose_resultado_detalhado


def test_compose_resultado_codigo_mais_camada() -> None:
    assert compose_resultado("10", "A") == "Classe 10 (camada A)"
    assert compose_resultado("102", "C") == "Classe 102 (camada C)"


def test_compose_resultado_revisao_manual_para_camada_d() -> None:
    assert compose_resultado("999", "D") == "Revisão manual"


def test_compose_resultado_detalhado_cita_peso_e_proba() -> None:
    text = compose_resultado_detalhado(
        classe_prevista="10",
        camada="A",
        weight=0.153795,
        probabilidade=0.91,
    )
    assert "camada A" in text
    assert "0.153795" in text
    assert "0.9100" in text


def test_compose_resultado_detalhado_fora_do_registry() -> None:
    text = compose_resultado_detalhado(
        classe_prevista="999",
        camada="D",
        weight=None,
        probabilidade=None,
    )
    assert "revisão manual" in text.lower()
    assert "fora do escopo" in text


class _FakePipeline:
    def transform(self, frame):
        return np.zeros((len(frame), 2))


class _FakeChampion:
    def __init__(self, label: int = 0):
        self._label = label

    def predict(self, x):
        return np.array([self._label] * len(x))

    def predict_proba(self, x):
        n = len(x)
        # duas classes: [0.8, 0.2]
        return np.tile(np.array([[0.8, 0.2]]), (n, 1))


def _fake_runtime(*, predicted_index: int = 0, include_class: bool = True) -> ModelRuntime:
    lookup = {}
    if include_class:
        lookup["10"] = ClassTierInfo(classe="10", tier="A", weight=0.15, count=100, percentage=40.0)
        lookup["1"] = ClassTierInfo(classe="1", tier="A", weight=0.2, count=80, percentage=30.0)
    return ModelRuntime(
        champion=_FakeChampion(predicted_index),
        preprocessing_pipeline=_FakePipeline(),
        target_classes=("10", "1"),
        class_lookup=lookup,
        discard_tier_label="D",
        retained_feature_columns=tuple(LaudoSinteticoRequest.model_fields.keys()),
        champion_metadata={"algorithm": "fake"},
        target_encoder=None,
    )


def test_analisar_laudo_sintetico_com_runtime_fake() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    response = analisar_laudo_sintetico(laudo, _fake_runtime(predicted_index=0))

    assert response.classe_prevista == "10"
    assert response.camada == "A"
    assert response.resultado == "Classe 10 (camada A)"
    assert response.predict_proba is not None
    assert response.predict_proba["10"] == pytest.approx(0.8)
    assert "0.150000" in response.resultado_detalhado
    assert "0.8000" in response.resultado_detalhado


def test_analisar_laudo_sintetico_classe_fora_do_registry_revisao_manual() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    response = analisar_laudo_sintetico(laudo, _fake_runtime(predicted_index=0, include_class=False))

    assert response.classe_prevista == "10"
    assert response.camada == "D"
    assert response.resultado == "Revisão manual"
    assert response.predict_proba is not None


def test_analisar_sem_runtime_levanta_model_runtime_error() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    with pytest.raises(ModelRuntimeError, match="indisponível"):
        analisar_laudo_sintetico(laudo, None)


def test_analisar_csv_upload_numera_linhas() -> None:
    example = LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    # CSV path usa LaudoCompletoRequest; aqui só exercitamos o service com payloads
    # que já têm as colunas retidas (o fake pipeline não valida schema completo).
    laudo = SimpleNamespace(**dict(example))
    laudo.model_dump = lambda: dict(example)  # type: ignore[method-assign]

    items = analisar_csv_upload([laudo, laudo], _fake_runtime())
    assert [item.numero_linha for item in items] == [1, 2]
    assert all(item.classe_prevista == "10" for item in items)


def test_load_model_runtime_real_artifacts() -> None:
    """Aceite de integração leve: artefatos reais em model/ após compile_models."""
    champion = Path("model/compiled/champion.pkl")
    if not champion.exists():
        pytest.skip("champion.pkl ausente — rode scripts/compile_models.py")

    runtime = load_model_runtime()
    assert len(runtime.target_classes) >= 2
    assert "10" in runtime.class_lookup or "1" in runtime.class_lookup
