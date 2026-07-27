"""Contratos Pydantic de entrada dos endpoints de inspeção de medidor (Task 006, §2).

Ver `api.schema_generation` para o racional de gerar `LaudoCompletoRequest`/
`LaudoSinteticoRequest` dinamicamente (via `pydantic.create_model`) em vez de
escrevê-los à mão — este módulo é o ponto em que essa geração é efetivamente
"amarrada" à API: lê o cache de especificação de campos
(`model/schemas/raw_dataset_field_spec.json`, gerado por
`scripts/generate_pydantic_schema.py`) e constrói os dois modelos a partir
dele, com fallback gracioso (lendo o dataset real diretamente, mais lento)
quando o cache ainda não existe.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .schema_generation import (
    FieldSpec,
    build_example_row,
    build_field_specs,
    create_request_model,
    load_field_specs,
)

from machine_learning.feature_engineering import (  # noqa: E402
    DEFAULT_TARGET_COLUMN,
    MANUALLY_REMOVED_FEATURES,
    load_dataset,
    resolve_dataset_path,
)

logger = logging.getLogger("imec_analysis.api.schema")

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
FIELD_SPEC_CACHE_PATH = MODEL_DIR / "schemas" / "raw_dataset_field_spec.json"


class InspecaoMedidorBase(BaseModel):
    """Base comum dos contratos de laudo (§2 do plano).

    ``extra="forbid"`` detecta, já na validação do payload (antes de qualquer
    processamento), campos enviados pelo cliente que não existem no schema
    real do dataset/modelo — evita, por exemplo, que um cliente confunda
    `LaudoCompletoRequest` com `LaudoSinteticoRequest` silenciosamente (o
    campo extra seria simplesmente ignorado sem essa configuração)."""

    model_config = ConfigDict(extra="forbid")


def _load_cached_field_specs() -> tuple[dict[str, FieldSpec], dict[str, object]] | None:
    if not FIELD_SPEC_CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(FIELD_SPEC_CACHE_PATH.read_text(encoding="utf-8"))
        return load_field_specs(payload["field_specs"]), dict(payload.get("example_row") or {})
    except (json.JSONDecodeError, KeyError, OSError):
        logger.warning(
            "Cache de schema em %s está corrompido/incompleto; recalculando a partir do dataset real.",
            FIELD_SPEC_CACHE_PATH,
        )
        return None


def _load_live_field_specs() -> tuple[dict[str, FieldSpec], dict[str, object]]:
    """Fallback lento (~10s): lê o Excel real diretamente, só acionado quando
    o cache ainda não foi gerado (ver `scripts/generate_pydantic_schema.py`)."""
    dataset_path = resolve_dataset_path(None)
    data_frame = load_dataset(dataset_path)
    field_specs = build_field_specs(data_frame, target_column=DEFAULT_TARGET_COLUMN)
    example_row = build_example_row(data_frame, field_specs)
    return field_specs, example_row


_MINIMAL_FALLBACK_FIELD_SPECS: dict[str, FieldSpec] = {
    "NUMLAUDO": FieldSpec(
        name="NUMLAUDO",
        python_type_name="str",
        nullable=True,
        null_pct=0.0,
        description=(
            "Placeholder — cache de schema indisponível e leitura do dataset real "
            "falhou. Rode 'python scripts/generate_pydantic_schema.py'."
        ),
    ),
}
"""Último recurso, usado apenas se nem o cache nem a leitura ao vivo do
dataset real estiverem disponíveis (ex.: ambiente sem o `.xlsx`) — garante
que a API ainda sobe (em modo degradado) em vez de falhar no import."""


def _resolve_field_specs() -> tuple[dict[str, FieldSpec], dict[str, object]]:
    cached = _load_cached_field_specs()
    if cached is not None:
        return cached

    logger.warning(
        "Cache de schema (%s) não encontrado; tentando ler o dataset real "
        "diretamente (lento). Rode 'python scripts/generate_pydantic_schema.py' "
        "para evitar isso em produção.",
        FIELD_SPEC_CACHE_PATH,
    )
    try:
        return _load_live_field_specs()
    except Exception:  # pragma: no cover - fallback de última instância
        logger.exception(
            "Não foi possível derivar o schema a partir do dataset real; usando "
            "placeholder mínimo. Os contratos de laudo ficarão degradados até "
            "que o cache seja gerado."
        )
        return dict(_MINIMAL_FALLBACK_FIELD_SPECS), {}


_FULL_FIELD_SPECS, _EXAMPLE_ROW = _resolve_field_specs()
_SINTETICO_FIELD_SPECS = {
    name: spec for name, spec in _FULL_FIELD_SPECS.items() if name not in MANUALLY_REMOVED_FEATURES
}
_SINTETICO_EXAMPLE_ROW = {key: value for key, value in _EXAMPLE_ROW.items() if key in _SINTETICO_FIELD_SPECS}


LaudoCompletoRequest = create_request_model(
    "LaudoCompletoRequest", _FULL_FIELD_SPECS, base=InspecaoMedidorBase, example=_EXAMPLE_ROW or None
)
"""Contrato "todas as features" (§2.1 do plano): espelha todas as colunas do
dataset real (`resultado_laudo_afericao.xlsx`), exceto o target
`CODRSTAFER`. Uso: consumidor que já possui o laudo completo e não quer se
preocupar em saber quais colunas o modelo de fato usa."""

LaudoSinteticoRequest = create_request_model(
    "LaudoSinteticoRequest", _SINTETICO_FIELD_SPECS, base=InspecaoMedidorBase, example=_SINTETICO_EXAMPLE_ROW or None
)
"""Contrato "features do modelo" (§2.2 do plano): mesma base do §2.1,
excluindo `MANUALLY_REMOVED_FEATURES` — o conjunto de colunas que o
`preprocessing_pipeline.pkl` de fato espera antes do drop heurístico/
constante/correlação (esses três aplicados automaticamente pelo pipeline,
não pelo contrato). Uso: integração mais "magra", em que o consumidor já
sabe filtrar o que é irrelevante para o modelo."""


def validate_laudo_completo_row(row: dict[str, object]) -> LaudoCompletoRequest:
    """Valida uma linha (dict) contra `LaudoCompletoRequest`.

    Usada pelo endpoint de upload CSV (§2.3): em vez de duplicar regras de
    validação para o cenário de lote, cada linha do CSV é validada
    reaproveitando o MESMO schema do laudo completo (ver plano, §2.3:
    "reaproveitar o mesmo schema Pydantic de 2.1 para validar cada linha")."""
    return LaudoCompletoRequest.model_validate(row)
