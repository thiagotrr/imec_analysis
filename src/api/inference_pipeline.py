"""Pipeline compartilhado de inferência (Task 007/008).

Fluxo único para laudo completo, sintético e cada linha de CSV:
payload → DataFrame (features retidas) → preprocessing → predict[/proba]
→ decode → lookup de camada → composição da resposta (template).

A revisão LLM (Task 008) é opcional e ocorre fora deste módulo, nos services
de laudo unitário — nunca no CSV em lote.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from llm.analysis import compose_resultado, compose_resultado_detalhado

from api.models.inspecao_request import RETAINED_FEATURE_COLUMNS
from api.model_runtime import ClassTierInfo, ModelRuntime, ModelRuntimeError

# Quando a probabilidade da classe prevista atinge esse limiar, `predict_proba`
# na response passa a trazer apenas essa classe (sem alternativas), para evitar
# ambiguidade. Configurável via env para ajuste fino sem novo deploy.
DEFAULT_PREDICT_PROBA_HIGH_CONFIDENCE_THRESHOLD = 0.90

# Quando a probabilidade da classe prevista fica abaixo do limiar de alta
# confiança, `predict_proba` traz no máximo esse número de classes (a classe
# prevista + as alternativas mais prováveis), em vez de toda a distribuição.
PREDICT_PROBA_LOW_CONFIDENCE_TOP_N = 3


def _high_confidence_threshold() -> float:
    raw = os.getenv("PREDICT_PROBA_CONFIDENCE_THRESHOLD")
    if raw is None or raw.strip() == "":
        return DEFAULT_PREDICT_PROBA_HIGH_CONFIDENCE_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_PREDICT_PROBA_HIGH_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class InferenceResult:
    numero_laudo: str | None
    classe_prevista: str
    camada: str
    resultado: str
    resultado_detalhado: str
    predict_proba: dict[str, float]
    revisao_llm: str | None = None


def _payload_to_mapping(payload: Any) -> Mapping[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, Mapping):
        return payload
    raise TypeError(f"Payload de inferência inválido: {type(payload)!r}")


def _extract_numero_laudo(data: Mapping[str, Any]) -> str | None:
    value = data.get("NUMLAUDO")
    if value is None:
        return None
    return str(value)


def _features_frame(data: Mapping[str, Any], columns: tuple[str, ...]) -> pd.DataFrame:
    missing = [column for column in columns if column not in data]
    if missing:
        raise ModelRuntimeError(
            "Payload sem colunas retidas obrigatórias para o pipeline: " + ", ".join(missing)
        )
    row = {column: data[column] for column in columns}
    return pd.DataFrame([row], columns=list(columns))


def _predict_proba_by_class(runtime: ModelRuntime, transformed: Any) -> dict[str, float]:
    if not hasattr(runtime.champion, "predict_proba"):
        return {}
    proba_matrix = np.asarray(runtime.champion.predict_proba(transformed))
    if proba_matrix.ndim != 2 or proba_matrix.shape[0] < 1:
        return {}
    row = proba_matrix[0]
    n = min(len(row), len(runtime.target_classes))
    return {
        runtime.target_classes[index]: float(row[index])
        for index in range(n)
    }


def _finalize_predict_proba(
    predict_proba: dict[str, float],
    classe_prevista: str,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Ordena `predict_proba` com a classe prevista primeiro e, quando sua
    probabilidade atinge o limiar de alta confiança, remove as demais classes.
    Abaixo do limiar, mantém no máximo as `PREDICT_PROBA_LOW_CONFIDENCE_TOP_N`
    classes mais prováveis (top-3), em vez de toda a distribuição.

    Fonte única: o dict resultante é usado tanto na response da API quanto no
    gate/prompt da revisão LLM (nenhuma alternativa é mostrada em nenhum lugar
    quando a confiança já é alta).
    """
    if not predict_proba:
        return {}

    resolved_threshold = _high_confidence_threshold() if threshold is None else threshold
    probabilidade_prevista = predict_proba.get(classe_prevista)
    if probabilidade_prevista is not None and probabilidade_prevista >= resolved_threshold:
        return {classe_prevista: probabilidade_prevista}

    others = sorted(
        (item for item in predict_proba.items() if item[0] != classe_prevista),
        key=lambda item: item[1],
        reverse=True,
    )
    remaining_slots = max(0, PREDICT_PROBA_LOW_CONFIDENCE_TOP_N - (1 if probabilidade_prevista is not None else 0))
    others = others[:remaining_slots]
    if probabilidade_prevista is None:
        return dict(others)
    return {classe_prevista: probabilidade_prevista, **dict(others)}


def infer_one(payload: Any, runtime: ModelRuntime) -> InferenceResult:
    """Executa uma inferência completa a partir de um laudo já validado."""
    data = _payload_to_mapping(payload)
    frame = _features_frame(data, runtime.retained_feature_columns or RETAINED_FEATURE_COLUMNS)

    try:
        transformed = runtime.preprocessing_pipeline.transform(frame)
        encoded = int(np.asarray(runtime.champion.predict(transformed)).ravel()[0])
        predict_proba = _predict_proba_by_class(runtime, transformed)
    except ModelRuntimeError:
        raise
    except Exception as exc:  # sklearn / booster errors → 500 no router
        raise ModelRuntimeError(f"Falha na inferência do modelo: {exc}") from exc

    classe_prevista = runtime.decode_prediction(encoded)
    tier_info: ClassTierInfo | None = runtime.lookup_class(classe_prevista)
    if tier_info is None:
        camada = runtime.discard_tier_label
        weight = None
    else:
        camada = tier_info.tier
        weight = tier_info.weight

    predict_proba = _finalize_predict_proba(predict_proba, classe_prevista)
    probabilidade_classe = predict_proba.get(classe_prevista)
    resultado = compose_resultado(classe_prevista, camada, runtime.discard_tier_label)
    resultado_detalhado = compose_resultado_detalhado(
        classe_prevista=classe_prevista,
        camada=camada,
        weight=weight,
        probabilidade=probabilidade_classe,
        discard_tier_label=runtime.discard_tier_label,
    )

    return InferenceResult(
        numero_laudo=_extract_numero_laudo(data),
        classe_prevista=classe_prevista,
        camada=camada,
        resultado=resultado,
        resultado_detalhado=resultado_detalhado,
        predict_proba=predict_proba,
        revisao_llm=None,
    )
