"""Services (camada de negócio) dos endpoints de inspeção de medidor (Task 006, §3/§4).

Esta task cobre APENAS contratos Pydantic + endpoints HTTP + compilação de
modelo (ver docs/task006_proximos_passos.md) — a inferência real (carregar o
`.pkl` compilado, pré-processar via `preprocessing_pipeline.pkl`, decodificar
via `target_encoder.pkl` e compor `resultado_detalhado` usando
`class_weight_registry.json`) é explicitamente **fora do escopo** (§4) e fica
para uma task futura de "services".

Os três services de análise abaixo (`analisar_laudo_completo`,
`analisar_laudo_sintetico`, `analisar_csv_upload`) são, portanto, stubs que
levantam `NotImplementedError` — o router (`inspecao_router.py`) converte essa
exceção em `HTTPException(500, ...)`, documentando o comportamento no
`description` de cada endpoint (ver §3 do plano).

`obter_info_modelos` é a EXCEÇÃO: como ela só lê metadados já persistidos em
`model/compiled/*.json` (gerados por `scripts/compile_models.py`/
`classification.model_compilation`), sem qualquer inferência, ela É
totalmente implementada nesta task.
"""
from __future__ import annotations

import json
from pathlib import Path

from machine_learning.classification.model_compilation import CHAMPION_STEM, get_compiled_model_dir
from machine_learning.data_preparation import DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME, get_model_dir

from .inspecao_response_model import ModeloInfoAlgoritmoResponse, ModeloInfoResponse

_SERVICE_NOT_IMPLEMENTED_TEMPLATE = (
    "Service de {contexto} ainda não implementado — ver docs/task006_proximos_passos.md, "
    "§4 ('Fora do escopo desta task'). Esta task cobre apenas contratos Pydantic, endpoints "
    "HTTP e compilação de modelo; a inferência real (carregar o .pkl compilado via "
    "model/compiled/, pré-processar com preprocessing_pipeline.pkl, decodificar com "
    "target_encoder.pkl e compor 'resultado_detalhado' a partir de class_weight_registry.json) "
    "fica para a próxima task."
)


def _not_implemented(contexto: str) -> None:
    raise NotImplementedError(_SERVICE_NOT_IMPLEMENTED_TEMPLATE.format(contexto=contexto))


def analisar_laudo_completo(laudo: object) -> object:
    """TODO(task de services): implementar a inferência real a partir de
    `LaudoCompletoRequest` — ver docs/task006_proximos_passos.md, §4."""
    _not_implemented("análise de laudo completo (POST /inspecao/laudo_completo)")


def analisar_laudo_sintetico(laudo: object) -> object:
    """TODO(task de services): implementar a inferência real a partir de
    `LaudoSinteticoRequest` — ver docs/task006_proximos_passos.md, §4."""
    _not_implemented("análise de laudo sintético (POST /inspecao/laudo_sintetico)")


def analisar_csv_upload(laudos: list[object]) -> list[object]:
    """TODO(task de services): implementar a inferência em lote a partir das
    linhas do CSV já validadas (`LaudoCompletoRequest.model_validate` por
    linha) — ver docs/task006_proximos_passos.md, §4."""
    _not_implemented("análise em lote via upload CSV (POST /inspecao/csv)")


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
    """Lê os metadados dos modelos compilados em `model/compiled/` (ver
    `classification.model_compilation`), sem carregar nenhum `.pkl` nem
    executar qualquer inferência — apenas leitura de arquivos `.json` já
    persistidos em disco por `scripts/compile_models.py`.

    Retorna uma resposta "vazia" (com `message` explicativo) em vez de
    lançar erro quando a compilação ainda não foi executada — a ausência de
    modelo compilado não é uma falha da API, é um estado válido (ainda) do
    ciclo de vida do projeto.
    """
    compiled_dir = get_compiled_model_dir(output_dir)
    champion_metadata = _read_json(compiled_dir / f"{CHAMPION_STEM}.json")

    other_metadata_paths = sorted(
        path
        for path in compiled_dir.glob("*.json")
        if path.stem != CHAMPION_STEM
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
