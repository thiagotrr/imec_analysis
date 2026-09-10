"""Testes unitários da camada de services / analysis / runtime (Task 007/008)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from api.models.inspecao_request import LaudoSinteticoRequest
from api.model_runtime import ClassTierInfo, ModelRuntime, ModelRuntimeError, load_model_runtime
from api.services.inspecao import analisar_csv_upload, analisar_laudo_sintetico
from llm.analysis import compose_resultado, compose_resultado_detalhado
from llm.config import LlmSettings
from llm.glossary import CodrstaferGlossary, GlossaryEntry


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
    def __init__(self, label: int = 0, proba_row: list[float] | None = None):
        self._label = label
        self._proba_row = proba_row or [0.8, 0.2]

    def predict(self, x):
        return np.array([self._label] * len(x))

    def predict_proba(self, x):
        n = len(x)
        return np.tile(np.array([self._proba_row]), (n, 1))


def _fake_runtime(
    *,
    predicted_index: int = 0,
    include_class: bool = True,
    tier: str = "A",
    proba_row: list[float] | None = None,
) -> ModelRuntime:
    lookup = {}
    if include_class:
        lookup["10"] = ClassTierInfo(classe="10", tier=tier, weight=0.15, count=100, percentage=40.0)
        lookup["1"] = ClassTierInfo(classe="1", tier="A", weight=0.2, count=80, percentage=30.0)
    return ModelRuntime(
        champion=_FakeChampion(predicted_index, proba_row=proba_row),
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
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0),
        llm_settings=LlmSettings(enabled=False),
    )

    assert response.classe_prevista == "10"
    assert response.camada == "A"
    assert response.resultado == "Classe 10 (camada A)"
    assert response.predict_proba is not None
    assert response.predict_proba["10"] == pytest.approx(0.8)
    assert "0.150000" in response.resultado_detalhado
    assert "0.8000" in response.resultado_detalhado
    assert response.revisao_llm is None


def test_analisar_laudo_sintetico_classe_fora_do_registry_revisao_manual() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0, include_class=False),
        llm_settings=LlmSettings(enabled=False),
    )

    assert response.classe_prevista == "10"
    assert response.camada == "D"
    assert response.resultado == "Revisão manual"
    assert response.predict_proba is not None


def test_analisar_laudo_sintetico_predict_proba_alta_confianca_trunca_para_classe_prevista() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0, proba_row=[0.95, 0.05]),
        llm_settings=LlmSettings(enabled=False),
    )

    assert response.classe_prevista == "10"
    assert response.predict_proba == {"10": pytest.approx(0.95)}


def test_analisar_laudo_sintetico_predict_proba_baixa_confianca_mantem_ordenacao_decrescente() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0, proba_row=[0.6, 0.4]),
        llm_settings=LlmSettings(enabled=False),
    )

    assert response.predict_proba is not None
    assert list(response.predict_proba.keys()) == ["10", "1"]
    assert response.predict_proba["10"] == pytest.approx(0.6)
    assert response.predict_proba["1"] == pytest.approx(0.4)


def test_analisar_laudo_sintetico_situacao_afericao_populada_com_glossario() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    glossary = CodrstaferGlossary(
        version="test",
        status="confirmado",
        entries={
            "10": GlossaryEntry(
                code="10",
                label="Reprovado",
                description="desc",
                situacao_codigo="R",
                situacao_label="Reprovado",
            )
        },
    )
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0),
        llm_settings=LlmSettings(enabled=False),
        glossary=glossary,
    )

    assert response.classe_prevista == "10"
    assert response.situacao_afericao == "Reprovado"


def test_analisar_laudo_sintetico_situacao_afericao_none_quando_classe_ausente_do_glossario() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    glossary = CodrstaferGlossary(version="test", status="confirmado", entries={})
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0),
        llm_settings=LlmSettings(enabled=False),
        glossary=glossary,
    )

    assert response.situacao_afericao is None


def test_analisar_laudo_sintetico_dsc_classe_prevista_populado_com_glossario() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    glossary = CodrstaferGlossary(
        version="test",
        status="confirmado",
        entries={
            "10": GlossaryEntry(
                code="10",
                label="Reprovado",
                description="Descritivo oficial da classe 10.",
                situacao_codigo="R",
                situacao_label="Reprovado",
            )
        },
    )
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0),
        llm_settings=LlmSettings(enabled=False),
        glossary=glossary,
    )

    assert response.dsc_classe_prevista == "Descritivo oficial da classe 10."


def test_analisar_laudo_sintetico_dsc_classe_prevista_none_quando_classe_ausente_do_glossario() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    glossary = CodrstaferGlossary(version="test", status="confirmado", entries={})
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0),
        llm_settings=LlmSettings(enabled=False),
        glossary=glossary,
    )

    assert response.dsc_classe_prevista is None


def test_analisar_laudo_sintetico_predict_proba_baixa_confianca_limita_top3_com_descritivos() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    glossary = CodrstaferGlossary(
        version="test",
        status="confirmado",
        entries={
            "10": GlossaryEntry(code="10", label="L10", description="Descritivo 10"),
            "1": GlossaryEntry(code="1", label="L1", description="Descritivo 1"),
        },
    )
    response = analisar_laudo_sintetico(
        laudo,
        _fake_runtime(predicted_index=0, proba_row=[0.6, 0.4]),
        llm_settings=LlmSettings(enabled=False),
        glossary=glossary,
    )

    assert response.predict_proba is not None
    assert len(response.predict_proba) <= 3
    assert response.dsc_predict_proba == {"10": "Descritivo 10", "1": "Descritivo 1"}


def test_analisar_laudo_sintetico_predict_proba_baixa_confianca_trunca_top3_de_muitas_classes() -> None:
    laudo = LaudoSinteticoRequest.model_validate(LaudoSinteticoRequest.model_config["json_schema_extra"]["example"])
    lookup = {
        "10": ClassTierInfo(classe="10", tier="A", weight=0.15, count=100, percentage=40.0),
        "1": ClassTierInfo(classe="1", tier="A", weight=0.2, count=80, percentage=30.0),
        "2": ClassTierInfo(classe="2", tier="B", weight=0.1, count=10, percentage=5.0),
        "3": ClassTierInfo(classe="3", tier="C", weight=0.05, count=5, percentage=2.0),
    }
    runtime = ModelRuntime(
        champion=_FakeChampion(0, proba_row=[0.4, 0.3, 0.2, 0.1]),
        preprocessing_pipeline=_FakePipeline(),
        target_classes=("10", "1", "2", "3"),
        class_lookup=lookup,
        discard_tier_label="D",
        retained_feature_columns=tuple(LaudoSinteticoRequest.model_fields.keys()),
        champion_metadata={"algorithm": "fake"},
        target_encoder=None,
    )
    response = analisar_laudo_sintetico(
        laudo,
        runtime,
        llm_settings=LlmSettings(enabled=False),
    )

    assert response.predict_proba is not None
    assert list(response.predict_proba.keys()) == ["10", "1", "2"]


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
    assert all(item.revisao_llm is None for item in items)


def test_load_model_runtime_real_artifacts() -> None:
    """Aceite de integração leve: artefatos reais em model/ após compile_models."""
    champion = Path("model/compiled/champion.pkl")
    if not champion.exists():
        pytest.skip("champion.pkl ausente — rode scripts/compile_models.py")

    runtime = load_model_runtime()
    assert len(runtime.target_classes) >= 2
    assert "10" in runtime.class_lookup or "1" in runtime.class_lookup
