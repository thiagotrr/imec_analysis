"""Contexto estruturado para revisão LLM pós-inferência (Task 008)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from machine_learning.data_preparation import get_model_dir

from api.inference_pipeline import InferenceResult
from api.model_runtime import ClassTierInfo, ModelRuntime

from .glossary import CodrstaferGlossary, empty_glossary

# Sinais clínicos do laudo enviados ao prompt (evitar PII / colunas irrelevantes).
FEATURE_SNAPSHOT_KEYS: tuple[str, ...] = (
    "INDRST_ENS_ISP",
    "INDRST_ENS_EXD",
    "INDRST_ENS_MRC",
    "INDRST_ENS_EXM",
    "INDRST_GER",
    "SIT_LACRE",
    "INDRST_ENS_CRP",
    "OBSAFER",
    "INDRST_ENS_ISP_1",
    "INDRST_ENS_EXD_1",
    "INDRST_ENS_MRC_1",
    "INDRST_ENS_EXM_1",
    "INDRST_GER_1",
    "INDRST_ENS_CRP_1",
)

DEFAULT_CLASSIFICATION_DETAILS_FILENAME = "classification/classification_details.json"
DEFAULT_TOP_K_PROBA = 5


@dataclass(frozen=True)
class ClassMetricsSnapshot:
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    support: int | None = None
    tier: str | None = None


@dataclass(frozen=True)
class AnalysisContext:
    """Pacote de evidências para o LLM — a predição permanece imutável."""

    numero_laudo: str | None
    classe_prevista: str
    camada: str
    weight: float | None
    probabilidade: float | None
    topk_proba: list[tuple[str, float]]
    top1_top2_gap: float | None
    class_metrics: ClassMetricsSnapshot | None
    feature_snapshot: dict[str, Any]
    resultado: str
    resultado_detalhado: str
    glossary_status: str = "empty"
    glossary_entries_text: str = ""
    discard_tier_label: str = "D"
    extra: dict[str, Any] = field(default_factory=dict)


def _topk_proba(predict_proba: Mapping[str, float], k: int = DEFAULT_TOP_K_PROBA) -> list[tuple[str, float]]:
    ordered = sorted(
        ((str(code), float(score)) for code, score in predict_proba.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ordered[: max(1, k)]


def _top1_top2_gap(topk: list[tuple[str, float]]) -> float | None:
    if len(topk) < 2:
        return None
    return float(topk[0][1] - topk[1][1])


def _feature_snapshot(data: Mapping[str, Any], keys: tuple[str, ...] = FEATURE_SNAPSHOT_KEYS) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in keys:
        if key in data:
            value = data[key]
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            # Evita textos longos de observação no prompt.
            if key == "OBSAFER" and len(text) > 280:
                text = text[:277] + "..."
            snapshot[key] = text if isinstance(value, str) else value
    return snapshot


def load_class_metrics_lookup(
    path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, ClassMetricsSnapshot]:
    """Lê métricas por classe de ``classification_details.json`` (fail-soft)."""
    resolved = (
        Path(path)
        if path is not None
        else get_model_dir(output_dir) / DEFAULT_CLASSIFICATION_DETAILS_FILENAME
    )
    if not resolved.exists():
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    # classification_details.json usa "results"; aceitar também "models" por compatibilidade.
    blocks = payload.get("results") or payload.get("models") or []
    if not isinstance(blocks, list) or not blocks:
        return {}

    # Preferir o primeiro bloco com per_class (tipicamente o campeão consolidado).
    per_class: list[Any] = []
    for model_block in blocks:
        if isinstance(model_block, Mapping) and model_block.get("per_class"):
            per_class = list(model_block["per_class"])
            break
    if not per_class and isinstance(blocks[0], Mapping):
        per_class = list(blocks[0].get("per_class") or [])

    lookup: dict[str, ClassMetricsSnapshot] = {}
    for entry in per_class:
        if not isinstance(entry, Mapping):
            continue
        code = entry.get("original_class")
        if code is None:
            continue
        lookup[str(code)] = ClassMetricsSnapshot(
            precision=float(entry["precision"]) if entry.get("precision") is not None else None,
            recall=float(entry["recall"]) if entry.get("recall") is not None else None,
            f1=float(entry["f1"]) if entry.get("f1") is not None else None,
            support=int(entry["support"]) if entry.get("support") is not None else None,
            tier=str(entry["tier"]) if entry.get("tier") is not None else None,
        )
    return lookup


def build_analysis_context(
    result: InferenceResult,
    payload: Mapping[str, Any] | Any,
    runtime: ModelRuntime,
    *,
    glossary: CodrstaferGlossary | None = None,
    class_metrics_lookup: Mapping[str, ClassMetricsSnapshot] | None = None,
    top_k: int = DEFAULT_TOP_K_PROBA,
) -> AnalysisContext:
    data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
    tier_info: ClassTierInfo | None = runtime.lookup_class(result.classe_prevista)
    weight = tier_info.weight if tier_info is not None else None
    probabilidade = result.predict_proba.get(result.classe_prevista)
    topk = _topk_proba(result.predict_proba, k=top_k)
    resolved_glossary = glossary or empty_glossary()
    codes_for_glossary = [result.classe_prevista] + [code for code, _ in topk if code != result.classe_prevista]
    metrics = None
    if class_metrics_lookup is not None:
        metrics = class_metrics_lookup.get(result.classe_prevista)

    return AnalysisContext(
        numero_laudo=result.numero_laudo,
        classe_prevista=result.classe_prevista,
        camada=result.camada,
        weight=weight,
        probabilidade=probabilidade,
        topk_proba=topk,
        top1_top2_gap=_top1_top2_gap(topk),
        class_metrics=metrics,
        feature_snapshot=_feature_snapshot(data),
        resultado=result.resultado,
        resultado_detalhado=result.resultado_detalhado,
        glossary_status=resolved_glossary.status,
        glossary_entries_text=resolved_glossary.prompt_block_for(codes_for_glossary),
        discard_tier_label=runtime.discard_tier_label,
    )
