"""Utilitário de geração automática dos contratos Pydantic de laudo (Task 006, §2).

DECISÃO DE DESIGN — geração dinâmica (`pydantic.create_model`) em vez de um
arquivo `.py` gerado: os modelos (`LaudoCompletoRequest`/`LaudoSinteticoRequest`,
ver ``src/api/inspecao_request_model.py``) são construídos EM RUNTIME a partir
de uma especificação de campos (``FieldSpec``) derivada de
``feature_engineering.build_dataset_profile`` sobre um ``pandas.DataFrame``.
Motivos:

1. **Nunca dessincronizar** (exigência do plano, §2.2): qualquer mudança nas
   colunas do dataset real (`resultado_laudo_afericao.xlsx`) se propaga
   automaticamente na próxima regeneração do schema — não há um arquivo
   `.py` gerado para lembrar de re-gerar/revisar manualmente a cada mudança
   de coluna.
2. **Testabilidade**: as funções abaixo recebem um ``DataFrame`` diretamente
   (real OU sintético), então os testes usam o MESMO dataset sintético de
   ``tests/conftest.py`` — rápido, sem tocar no Excel real de 35k linhas —
   para validar a lógica de inferência de tipo/nulidade; o comportamento é
   idêntico ao usado com dados reais.
3. **Simplicidade**: evita um passo de "geração de código" que precisaria
   ser versionado, revisado em PR e mantido manualmente em sincronia;
   ``create_model`` resolve isso em poucas linhas, sem arquivo intermediário.

Uso em produção (API): reconstruir o ``DatasetProfile`` a partir do Excel real
a cada import do módulo da API seria lento (leitura de ~35k linhas a cada
``uvicorn --reload``) e desnecessário (o formato do dataset não muda a cada
request). Por isso, ``scripts/generate_pydantic_schema.py`` gera, UMA VEZ
(offline, quando o dataset/colunas mudarem), um cache leve em
``model/schemas/raw_dataset_field_spec.json`` com a especificação de campos
(nome/tipo/nulidade/exemplo) já calculada. A API
(``src/api/inspecao_request_model.py``) apenas LÊ esse cache (JSON pequeno,
leitura instantânea) para reconstruir os mesmos modelos via ``create_model``
— sem depender do Excel em tempo de execução (com fallback explícito para o
caminho lento caso o cache ainda não exista, ver esse módulo).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from machine_learning.feature_engineering import (  # noqa: E402
    DEFAULT_TARGET_COLUMN,
    MANUALLY_REMOVED_FEATURES,
    build_dataset_profile,
)


_PYTHON_TYPES_BY_NAME: dict[str, type] = {"int": int, "float": float, "bool": bool, "str": str}


@dataclass(frozen=True)
class FieldSpec:
    """Especificação mínima de um campo, suficiente para montar um `Field` Pydantic."""

    name: str
    python_type_name: str
    """Um de ``_PYTHON_TYPES_BY_NAME`` — guardamos o NOME (não o `type` em si)
    para que a especificação seja serializável em JSON (ver `to_dict`/`from_dict`,
    usadas pelo cache de `scripts/generate_pydantic_schema.py`)."""
    nullable: bool
    null_pct: float
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "python_type_name": self.python_type_name,
            "nullable": self.nullable,
            "null_pct": self.null_pct,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FieldSpec":
        return cls(
            name=str(payload["name"]),
            python_type_name=str(payload["python_type_name"]),
            nullable=bool(payload["nullable"]),
            null_pct=float(payload.get("null_pct", 0.0)),
            description=str(payload.get("description", "")),
        )


def _infer_python_type_name(dtype: object) -> str:
    """Mapeia um dtype `pandas` para um tipo Python simples (Field do contrato).

    Simplificação deliberada: colunas de texto livre, categóricas e datetime
    são todas tratadas como ``str`` no contrato de ENTRADA — a normalização
    fina de formato (ex.: parse de datas) é responsabilidade do futuro
    service de inferência (fora do escopo desta task, ver
    docs/task006_proximos_passos.md, §4), não do contrato de validação de
    schema em si.
    """
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "str"


def build_field_specs(
    data_frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    excluded_columns: Sequence[str] = (),
) -> dict[str, FieldSpec]:
    """Deriva a especificação de campos de ``data_frame`` (todas as colunas exceto ``target_column``/``excluded_columns``).

    ``nullable`` é decidido pelo ``null_pct`` real da coluna (``DatasetProfile.null_pct``,
    ver `feature_engineering.build_dataset_profile`): colunas sem nenhum valor
    faltante no dataset observado são obrigatórias no contrato; as demais são
    opcionais (``default=None``) — replica a tolerância a nulos real do
    dataset (§2.4 do plano), em vez de uma decisão manual/arbitrária.
    """
    profile = build_dataset_profile(data_frame)
    excluded = {target_column, *excluded_columns}
    field_specs: dict[str, FieldSpec] = {}
    for column in data_frame.columns:
        if column in excluded:
            continue
        dtype = profile.types_by_col[column]
        null_pct = float(profile.null_pct.get(column, 0.0))
        field_specs[column] = FieldSpec(
            name=column,
            python_type_name=_infer_python_type_name(dtype),
            nullable=null_pct > 0,
            null_pct=null_pct,
            description=f"Coluna '{column}' do laudo de aferição (dtype original: {dtype}; nulos observados: {null_pct}%).",
        )
    return field_specs


def build_laudo_sintetico_field_specs(
    data_frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    excluded_columns: Sequence[str] = MANUALLY_REMOVED_FEATURES,
) -> dict[str, FieldSpec]:
    """Especificação do contrato "features do modelo" (§2.2): mesma base do
    §2.1, excluindo ``MANUALLY_REMOVED_FEATURES`` — usa o MESMO
    `build_field_specs`, garantindo que os dois contratos nunca dessincronizem
    manualmente (a única diferença é o parâmetro `excluded_columns`)."""
    return build_field_specs(data_frame, target_column=target_column, excluded_columns=excluded_columns)


def create_request_model(
    model_name: str,
    field_specs: dict[str, FieldSpec] | Sequence[FieldSpec],
    base: type[BaseModel],
    example: dict[str, object] | None = None,
) -> type[BaseModel]:
    """Monta um `BaseModel` via `pydantic.create_model` a partir de `field_specs`.

    `example` (quando informado) é anexado a `model_config.json_schema_extra`
    — não é possível passar `__config__` junto de `__base__` em
    `create_model` (mutuamente exclusivos), então o `model_config` é
    sobrescrito e reconstruído (`model_rebuild(force=True)`) DEPOIS da
    criação da classe, herdando as demais opções (`extra="forbid"`) de
    `base.model_config`.
    """
    specs = field_specs.values() if isinstance(field_specs, dict) else field_specs
    field_definitions: dict[str, tuple[type, object]] = {}
    for spec in specs:
        python_type = _PYTHON_TYPES_BY_NAME[spec.python_type_name]
        if spec.nullable:
            field_definitions[spec.name] = (python_type | None, Field(default=None, description=spec.description))
        else:
            field_definitions[spec.name] = (python_type, Field(..., description=spec.description))

    model = create_model(model_name, __base__=base, **field_definitions)
    if example is not None:
        merged_config = {**dict(base.model_config), "json_schema_extra": {"example": example}}
        model.model_config = ConfigDict(**merged_config)
        model.model_rebuild(force=True)
    return model


def dump_field_specs(field_specs: dict[str, FieldSpec]) -> dict[str, dict[str, object]]:
    return {name: spec.to_dict() for name, spec in field_specs.items()}


def load_field_specs(payload: dict[str, dict[str, object]]) -> dict[str, FieldSpec]:
    return {name: FieldSpec.from_dict(spec) for name, spec in payload.items()}


def pick_example_row_index(data_frame: pd.DataFrame) -> int:
    """Escolhe a linha com menos valores nulos — exemplo mais "completo" e
    ilustrativo para o `json_schema_extra` dos contratos (ver §2.4 do plano)."""
    if data_frame.empty:
        raise ValueError("Não é possível escolher uma linha de exemplo de um DataFrame vazio.")
    return int(data_frame.isna().sum(axis=1).idxmin())


def build_example_row(
    data_frame: pd.DataFrame,
    field_specs: dict[str, FieldSpec],
    row_index: int | None = None,
) -> dict[str, object]:
    """Converte uma linha real de `data_frame` em um dict JSON-serializável,
    restrito às colunas de `field_specs` — usado como `json_schema_extra["example"]`."""
    resolved_index = row_index if row_index is not None else pick_example_row_index(data_frame)
    row = data_frame.iloc[resolved_index]
    example: dict[str, object] = {}
    for name, spec in field_specs.items():
        if name not in row.index:
            continue
        value = row[name]
        if pd.isna(value):
            example[name] = None
            continue
        python_type = _PYTHON_TYPES_BY_NAME[spec.python_type_name]
        try:
            example[name] = python_type(value)
        except (TypeError, ValueError):
            example[name] = str(value)
    return example
