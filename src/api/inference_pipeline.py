"""Pipeline compartilhado de inferência (Task 007).

Fluxo único para laudo completo, sintético e cada linha de CSV:
payload → DataFrame (features retidas) → preprocessing → predict[/proba]
→ decode → lookup de camada → composição da resposta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .inspecao_request_model import RETAINED_FEATURE_COLUMNS
from .model_runtime import ClassTierInfo, ModelRuntime, ModelRuntimeError
from .narrative import compose_resultado, compose_resultado_detalhado


@dataclass(frozen=True)
class InferenceResult:
    numero_laudo: str | None
    classe_prevista: str
    camada: str
    resultado: str
    resultado_detalhado: str
    predict_proba: dict[str, float]


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
    )
