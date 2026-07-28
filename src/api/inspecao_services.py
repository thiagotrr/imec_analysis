"""Services de inspeção de medidor (Task 007 — inferência real).

Carrega artefatos via ``ModelRuntime`` (startup) e executa o pipeline único
em ``inference_pipeline.infer_one``. ``GET /inspecao/modelos`` continua sendo
apenas leitura de metadados JSON (sem inferência).
"""
from __future__ import annotations

import json
from pathlib import Path

from machine_learning.classification.model_compilation import CHAMPION_STEM, get_compiled_model_dir
from machine_learning.data_preparation import DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME, get_model_dir

from .inference_pipeline import InferenceResult, infer_one
from .inspecao_response_model import (
    InspecaoLaudoCsvItemResponse,
    InspecaoLaudoResponse,
    ModeloInfoAlgoritmoResponse,
    ModeloInfoResponse,
)
from .model_runtime import ModelRuntime, ModelRuntimeError


def _require_runtime(runtime: ModelRuntime | None) -> ModelRuntime:
    if runtime is None:
        raise ModelRuntimeError(
            "Runtime de inferência indisponível: artefatos não carregados no startup. "
            "Rode 'python scripts/compile_models.py' e reinicie a API."
        )
    return runtime


def _to_response(result: InferenceResult) -> InspecaoLaudoResponse:
    return InspecaoLaudoResponse(
        numero_laudo=result.numero_laudo,
        classe_prevista=result.classe_prevista,
        camada=result.camada,
        resultado=result.resultado,
        resultado_detalhado=result.resultado_detalhado,
        predict_proba=result.predict_proba or None,
    )


def analisar_laudo_completo(laudo: object, runtime: ModelRuntime | None) -> InspecaoLaudoResponse:
    return _to_response(infer_one(laudo, _require_runtime(runtime)))


def analisar_laudo_sintetico(laudo: object, runtime: ModelRuntime | None) -> InspecaoLaudoResponse:
    return _to_response(infer_one(laudo, _require_runtime(runtime)))


def analisar_csv_upload(
    laudos: list[object],
    runtime: ModelRuntime | None,
) -> list[InspecaoLaudoCsvItemResponse]:
    resolved = _require_runtime(runtime)
    items: list[InspecaoLaudoCsvItemResponse] = []
    for line_number, laudo in enumerate(laudos, start=1):
        result = infer_one(laudo, resolved)
        base = _to_response(result)
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
