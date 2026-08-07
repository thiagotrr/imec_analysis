"""Services de inspeção de medidor (Task 007/008 — inferência + revisão LLM opcional).

Carrega artefatos via ``ModelRuntime`` (startup) e executa o pipeline único
em ``inference_pipeline.infer_one``. A revisão LLM (LangChain) só entra nos
endpoints unitários, com gate condicional e fail-soft.
``GET /inspecao/modelos`` continua sendo apenas leitura de metadados JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from machine_learning.classification.model_compilation import CHAMPION_STEM, get_compiled_model_dir
from machine_learning.data_preparation import DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME, get_model_dir

from llm.config import LlmSettings, load_llm_settings
from llm.context import (
    ClassMetricsSnapshot,
    build_analysis_context,
    load_class_metrics_lookup,
)
from llm.glossary import CodrstaferGlossary, empty_glossary, load_codrstafer_glossary
from llm.reviewer import SupportsReview, generate_revisao_llm

from api.inference_pipeline import InferenceResult, infer_one
from api.models.inspecao_response import (
    InspecaoLaudoCsvItemResponse,
    InspecaoLaudoResponse,
    ModeloInfoAlgoritmoResponse,
    ModeloInfoResponse,
)
from api.model_runtime import ModelRuntime, ModelRuntimeError


def _require_runtime(runtime: ModelRuntime | None) -> ModelRuntime:
    if runtime is None:
        raise ModelRuntimeError(
            "Runtime de inferência indisponível: artefatos não carregados no startup. "
            "Rode 'python scripts/compile_models.py' e reinicie a API."
        )
    return runtime


def _situacao_afericao(classe_prevista: str, glossary: CodrstaferGlossary | None) -> str | None:
    if glossary is None:
        return None
    entry = glossary.get(classe_prevista)
    return entry.situacao_label if entry is not None else None


def _to_response(
    result: InferenceResult,
    *,
    glossary: CodrstaferGlossary | None = None,
) -> InspecaoLaudoResponse:
    return InspecaoLaudoResponse(
        numero_laudo=result.numero_laudo,
        classe_prevista=result.classe_prevista,
        camada=result.camada,
        situacao_afericao=_situacao_afericao(result.classe_prevista, glossary),
        resultado=result.resultado,
        resultado_detalhado=result.resultado_detalhado,
        predict_proba=result.predict_proba or None,
        revisao_llm=result.revisao_llm,
    )


def _with_revisao(result: InferenceResult, revisao_llm: str | None) -> InferenceResult:
    return InferenceResult(
        numero_laudo=result.numero_laudo,
        classe_prevista=result.classe_prevista,
        camada=result.camada,
        resultado=result.resultado,
        resultado_detalhado=result.resultado_detalhado,
        predict_proba=result.predict_proba,
        revisao_llm=revisao_llm,
    )


def _attach_llm_review(
    result: InferenceResult,
    payload: object,
    runtime: ModelRuntime,
    *,
    force_revisao_llm: bool | None = None,
    llm_reviewer: SupportsReview | None = None,
    llm_settings: LlmSettings | None = None,
    glossary: CodrstaferGlossary | None = None,
    class_metrics_lookup: dict[str, ClassMetricsSnapshot] | None = None,
) -> InferenceResult:
    """Pós-processamento opcional: nunca altera predição/camada/proba."""
    settings = llm_settings or load_llm_settings()
    context = build_analysis_context(
        result,
        payload,
        runtime,
        glossary=glossary,
        class_metrics_lookup=class_metrics_lookup,
    )
    revisao, _decision = generate_revisao_llm(
        context,
        llm_reviewer,
        settings,
        force=force_revisao_llm,
    )
    return _with_revisao(result, revisao)


def analisar_laudo_completo(
    laudo: object,
    runtime: ModelRuntime | None,
    *,
    revisao_llm: bool | None = None,
    llm_reviewer: SupportsReview | None = None,
    llm_settings: LlmSettings | None = None,
    glossary: CodrstaferGlossary | None = None,
    class_metrics_lookup: dict[str, ClassMetricsSnapshot] | None = None,
) -> InspecaoLaudoResponse:
    resolved = _require_runtime(runtime)
    result = infer_one(laudo, resolved)
    resolved_glossary = glossary if glossary is not None else load_codrstafer_glossary()
    enriched = _attach_llm_review(
        result,
        laudo,
        resolved,
        force_revisao_llm=revisao_llm,
        llm_reviewer=llm_reviewer,
        llm_settings=llm_settings,
        glossary=resolved_glossary,
        class_metrics_lookup=(
            class_metrics_lookup
            if class_metrics_lookup is not None
            else load_class_metrics_lookup()
        ),
    )
    return _to_response(enriched, glossary=resolved_glossary)


def analisar_laudo_sintetico(
    laudo: object,
    runtime: ModelRuntime | None,
    *,
    revisao_llm: bool | None = None,
    llm_reviewer: SupportsReview | None = None,
    llm_settings: LlmSettings | None = None,
    glossary: CodrstaferGlossary | None = None,
    class_metrics_lookup: dict[str, ClassMetricsSnapshot] | None = None,
) -> InspecaoLaudoResponse:
    resolved = _require_runtime(runtime)
    result = infer_one(laudo, resolved)
    resolved_glossary = glossary if glossary is not None else load_codrstafer_glossary()
    enriched = _attach_llm_review(
        result,
        laudo,
        resolved,
        force_revisao_llm=revisao_llm,
        llm_reviewer=llm_reviewer,
        llm_settings=llm_settings,
        glossary=resolved_glossary,
        class_metrics_lookup=(
            class_metrics_lookup
            if class_metrics_lookup is not None
            else load_class_metrics_lookup()
        ),
    )
    return _to_response(enriched, glossary=resolved_glossary)


def analisar_csv_upload(
    laudos: list[object],
    runtime: ModelRuntime | None,
    *,
    glossary: CodrstaferGlossary | None = None,
) -> list[InspecaoLaudoCsvItemResponse]:
    """CSV em lote: apenas template (sem LLM linha a linha)."""
    resolved = _require_runtime(runtime)
    resolved_glossary = glossary if glossary is not None else load_codrstafer_glossary()
    items: list[InspecaoLaudoCsvItemResponse] = []
    for line_number, laudo in enumerate(laudos, start=1):
        result = infer_one(laudo, resolved)
        base = _to_response(result, glossary=resolved_glossary)
        items.append(
            InspecaoLaudoCsvItemResponse(
                **base.model_dump(),
                numero_linha=line_number,
            )
        )
    return items


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _metadata_to_response(metadata: dict[str, object]) -> ModeloInfoAlgoritmoResponse:
    feature_columns = metadata.get("feature_columns") or []
    return ModeloInfoAlgoritmoResponse(
        algorithm=str(metadata.get("algorithm", "desconhecido")),
        resampling=str(metadata.get("resampling", "none")),
        trained_at=metadata.get("trained_at"),
        metrics=dict(metadata.get("metrics") or {}),
        feature_columns_count=len(feature_columns) if feature_columns else None,
        target_classes=metadata.get("target_classes"),
    )


def obter_info_modelos(output_dir: str | Path | None = None) -> ModeloInfoResponse:
    """Lê metadados em ``model/compiled/*.json`` sem carregar `.pkl` nem inferir."""
    compiled_dir = get_compiled_model_dir(output_dir)
    champion_metadata = _read_json(compiled_dir / f"{CHAMPION_STEM}.json")

    other_metadata_paths = sorted(
        path for path in compiled_dir.glob("*.json") if path.stem != CHAMPION_STEM
    )
    compiled_variants = []
    for path in other_metadata_paths:
        metadata = _read_json(path)
        if metadata is not None:
            compiled_variants.append(_metadata_to_response(metadata))

    class_weight_registry_path = get_model_dir(output_dir) / DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME

    if champion_metadata is None:
        return ModeloInfoResponse(
            champion=None,
            compiled_variants=compiled_variants,
            tier_thresholds=None,
            class_weight_registry_available=class_weight_registry_path.exists(),
            message=(
                "Nenhum modelo compilado encontrado em "
                f"{compiled_dir}. Rode 'python scripts/compile_models.py' primeiro."
            ),
        )

    return ModeloInfoResponse(
        champion=_metadata_to_response(champion_metadata),
        compiled_variants=compiled_variants,
        tier_thresholds=champion_metadata.get("tier_thresholds"),
        class_weight_registry_available=class_weight_registry_path.exists(),
        message=None,
    )


def load_llm_dependencies(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Carrega glossário, métricas e settings LLM no startup (fail-soft)."""
    settings = load_llm_settings()
    glossary = load_codrstafer_glossary(output_dir=output_dir)
    if glossary.status == "empty":
        glossary = empty_glossary()
    return {
        "llm_settings": settings,
        "glossary": glossary,
        "class_metrics_lookup": load_class_metrics_lookup(output_dir=output_dir),
    }
