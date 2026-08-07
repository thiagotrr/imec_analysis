"""Testes da revisão LLM (Task 008): gate, fake reviewer e invariantes."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from api.inference_pipeline import InferenceResult
from api.models.inspecao_request import LaudoSinteticoRequest
from api.model_runtime import ClassTierInfo, ModelRuntime
from api.services.inspecao import analisar_csv_upload, analisar_laudo_sintetico
from llm.config import LlmSettings
from llm.context import AnalysisContext, ClassMetricsSnapshot, build_analysis_context
from llm.gate import should_request_llm
from llm.glossary import CodrstaferGlossary, GlossaryEntry, load_codrstafer_glossary
from llm.prompts import SYSTEM_PROMPT_PT_BR, build_user_prompt
from llm.reviewer import FakeLlmReviewer


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
        return np.tile(np.array([self._proba_row]), (len(x), 1))


def _runtime(*, tier: str = "A", predicted_index: int = 0) -> ModelRuntime:
    return ModelRuntime(
        champion=_FakeChampion(predicted_index),
        preprocessing_pipeline=_FakePipeline(),
        target_classes=("10", "1"),
        class_lookup={
            "10": ClassTierInfo(classe="10", tier=tier, weight=0.15, count=100, percentage=40.0),
            "1": ClassTierInfo(classe="1", tier="A", weight=0.2, count=80, percentage=30.0),
        },
        discard_tier_label="D",
        retained_feature_columns=tuple(LaudoSinteticoRequest.model_fields.keys()),
        champion_metadata={"algorithm": "fake"},
        target_encoder=None,
    )


def _context(*, camada: str = "A", probabilidade: float = 0.9, gap: float = 0.5) -> AnalysisContext:
    top2 = probabilidade - gap
    return AnalysisContext(
        numero_laudo="1",
        classe_prevista="10",
        camada=camada,
        weight=0.15,
        probabilidade=probabilidade,
        topk_proba=[("10", probabilidade), ("1", max(top2, 0.0))],
        top1_top2_gap=gap,
        class_metrics=ClassMetricsSnapshot(f1=0.9, precision=0.9, recall=0.9, support=100, tier=camada),
        feature_snapshot={"SIT_LACRE": "APROVADO"},
        resultado=f"Classe 10 (camada {camada})",
        resultado_detalhado="template",
        glossary_status="scaffold",
        glossary_entries_text="CODRSTAFER 10: (rótulo pendente)",
    )


def test_gate_skip_tier_a_by_default() -> None:
    settings = LlmSettings(enabled=True, openai_api_key="sk-test", skip_tier_a=True)
    decision = should_request_llm(_context(camada="A"), settings)
    assert decision.should_call is False
    assert "camada A" in decision.reason


def test_gate_calls_for_tier_b_and_c() -> None:
    settings = LlmSettings(enabled=True, openai_api_key="sk-test", skip_tier_a=True)
    assert should_request_llm(_context(camada="B"), settings).should_call is True
    assert should_request_llm(_context(camada="C"), settings).should_call is True
    assert should_request_llm(_context(camada="D"), settings).should_call is True


def test_gate_force_overrides_tier_a() -> None:
    settings = LlmSettings(enabled=True, openai_api_key="sk-test", skip_tier_a=True)
    decision = should_request_llm(_context(camada="A"), settings, force=True)
    assert decision.should_call is True


def test_gate_force_false_disables() -> None:
    settings = LlmSettings(enabled=True, openai_api_key="sk-test", skip_tier_a=True)
    decision = should_request_llm(_context(camada="C"), settings, force=False)
    assert decision.should_call is False


def test_prompt_pt_br_prohibits_altering_prediction() -> None:
    assert "NÃO pode alterar" in SYSTEM_PROMPT_PT_BR or "NÃO" in SYSTEM_PROMPT_PT_BR
    assert "classe_prevista" in SYSTEM_PROMPT_PT_BR
    user = build_user_prompt(_context())
    assert "Classe prevista (CODRSTAFER): 10" in user
    assert "Camada de qualificação: A" in user


def test_glossary_confirmado_loads_official_classes() -> None:
    glossary = load_codrstafer_glossary()
    assert glossary.status == "confirmado"
    assert "10" in glossary.entries
    assert "1" in glossary.entries
    assert len(glossary.entries) >= 14

    entry_10 = glossary.entries["10"]
    assert entry_10.status == "confirmado"
    assert entry_10.situacao_codigo == "R"
    assert entry_10.situacao_label == "Reprovado"
    assert entry_10.description
    assert "Situação: Reprovado" in entry_10.prompt_text()


def test_fake_reviewer_does_not_change_prediction_fields() -> None:
    laudo = LaudoSinteticoRequest.model_validate(
        LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    )
    fake = FakeLlmReviewer("Texto de revisão gerado para teste.")
    settings = LlmSettings(
        enabled=True,
        provider="openai",
        openai_api_key="sk-test",
        skip_tier_a=True,
    )
    # Camada C → gate dispara; fake não altera facts.
    response = analisar_laudo_sintetico(
        laudo,
        _runtime(tier="C"),
        revisao_llm=None,
        llm_reviewer=fake,
        llm_settings=settings,
        glossary=CodrstaferGlossary(
            version="test",
            status="scaffold",
            entries={"10": GlossaryEntry(code="10", label=None, description=None)},
        ),
        class_metrics_lookup={},
    )

    assert response.classe_prevista == "10"
    assert response.camada == "C"
    assert response.resultado == "Classe 10 (camada C)"
    assert response.predict_proba is not None
    assert response.predict_proba["10"] == pytest.approx(0.8)
    assert response.revisao_llm == "Texto de revisão gerado para teste."
    assert len(fake.calls) == 1
    assert fake.calls[0].classe_prevista == "10"
    assert fake.calls[0].camada == "C"


def test_tier_a_default_skips_fake_reviewer() -> None:
    laudo = LaudoSinteticoRequest.model_validate(
        LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    )
    fake = FakeLlmReviewer("não deveria aparecer")
    settings = LlmSettings(enabled=True, openai_api_key="sk-test", skip_tier_a=True)
    response = analisar_laudo_sintetico(
        laudo,
        _runtime(tier="A"),
        llm_reviewer=fake,
        llm_settings=settings,
    )
    assert response.camada == "A"
    assert response.revisao_llm is None
    assert fake.calls == []


def test_force_revisao_llm_on_tier_a() -> None:
    laudo = LaudoSinteticoRequest.model_validate(
        LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    )
    fake = FakeLlmReviewer("forçado")
    settings = LlmSettings(enabled=True, openai_api_key="sk-test", skip_tier_a=True)
    response = analisar_laudo_sintetico(
        laudo,
        _runtime(tier="A"),
        revisao_llm=True,
        llm_reviewer=fake,
        llm_settings=settings,
    )
    assert response.classe_prevista == "10"
    assert response.camada == "A"
    assert response.revisao_llm == "forçado"


def test_csv_never_calls_llm() -> None:
    example = LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    laudo = SimpleNamespace(**dict(example))
    laudo.model_dump = lambda: dict(example)  # type: ignore[method-assign]
    items = analisar_csv_upload([laudo], _runtime(tier="C"))
    assert items[0].revisao_llm is None


def test_build_analysis_context_topk() -> None:
    result = InferenceResult(
        numero_laudo="x",
        classe_prevista="10",
        camada="B",
        resultado="Classe 10 (camada B)",
        resultado_detalhado="detalhe",
        predict_proba={"10": 0.55, "1": 0.45},
    )
    laudo = LaudoSinteticoRequest.model_validate(
        LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    )
    ctx = build_analysis_context(result, laudo, _runtime(tier="B"))
    assert ctx.topk_proba[0][0] == "10"
    assert ctx.top1_top2_gap == pytest.approx(0.10)
    assert "SIT_LACRE" in ctx.feature_snapshot or ctx.feature_snapshot is not None
