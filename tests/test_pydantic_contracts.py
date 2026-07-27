from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from api.inspecao_request_model import (
    RETAINED_FEATURE_COLUMNS,
    InspecaoMedidorBase,
    LaudoCompletoRequest,
    LaudoSinteticoRequest,
    validate_laudo_completo_row,
)
from api.schema_generation import (
    build_example_row,
    build_field_specs,
    build_laudo_sintetico_field_specs,
    create_request_model,
    dump_field_specs,
    load_field_specs,
)
from machine_learning.feature_engineering import MANUALLY_REMOVED_FEATURES

from conftest import TARGET_COLUMN


# ---------------------------------------------------------------------------
# `api.schema_generation` — testado com o dataset SINTÉTICO (rápido, sem
# tocar no Excel real de 35k linhas), conforme exigido pelas restrições da task.
# ---------------------------------------------------------------------------


def test_build_field_specs_excludes_target_column(synthetic_dataset: pd.DataFrame) -> None:
    field_specs = build_field_specs(synthetic_dataset, target_column=TARGET_COLUMN)

    assert TARGET_COLUMN not in field_specs
    assert set(field_specs) == set(synthetic_dataset.columns) - {TARGET_COLUMN}


def test_build_field_specs_infers_types_from_dtype(synthetic_dataset: pd.DataFrame) -> None:
    field_specs = build_field_specs(synthetic_dataset, target_column=TARGET_COLUMN)

    assert field_specs["NUMERIC_FEATURE_1"].python_type_name == "float"
    assert field_specs["CATEGORICAL_FEATURE"].python_type_name == "str"


def test_build_field_specs_nullable_matches_observed_null_percentage(synthetic_dataset: pd.DataFrame) -> None:
    # Nenhuma coluna do dataset sintético tem nulos por construção — todas
    # devem ser marcadas como obrigatórias (nullable=False).
    field_specs = build_field_specs(synthetic_dataset, target_column=TARGET_COLUMN)
    assert all(not spec.nullable for spec in field_specs.values())

    # Introduzindo nulos artificialmente, a coluna correspondente deve virar opcional.
    dataset_with_nulls = synthetic_dataset.copy()
    dataset_with_nulls.loc[0, "NUMERIC_FEATURE_1"] = None
    field_specs_with_nulls = build_field_specs(dataset_with_nulls, target_column=TARGET_COLUMN)
    assert field_specs_with_nulls["NUMERIC_FEATURE_1"].nullable is True
    assert field_specs_with_nulls["NUMERIC_FEATURE_1"].null_pct > 0


def test_build_laudo_sintetico_field_specs_excludes_manually_removed_features(
    synthetic_dataset: pd.DataFrame,
) -> None:
    dataset_with_removed_feature = synthetic_dataset.copy()
    dataset_with_removed_feature[MANUALLY_REMOVED_FEATURES[0]] = 1

    full_specs = build_field_specs(dataset_with_removed_feature, target_column=TARGET_COLUMN)
    sintetico_specs = build_laudo_sintetico_field_specs(dataset_with_removed_feature, target_column=TARGET_COLUMN)

    assert MANUALLY_REMOVED_FEATURES[0] in full_specs
    assert MANUALLY_REMOVED_FEATURES[0] not in sintetico_specs
    # As demais colunas (não removidas manualmente) devem ser idênticas nos dois contratos.
    assert set(full_specs) - {MANUALLY_REMOVED_FEATURES[0]} == set(sintetico_specs)


def test_dump_and_load_field_specs_roundtrip(synthetic_dataset: pd.DataFrame) -> None:
    field_specs = build_field_specs(synthetic_dataset, target_column=TARGET_COLUMN)
    round_tripped = load_field_specs(dump_field_specs(field_specs))

    assert round_tripped == field_specs


def test_create_request_model_forbids_extra_fields_and_enforces_required(synthetic_dataset: pd.DataFrame) -> None:
    field_specs = build_field_specs(synthetic_dataset, target_column=TARGET_COLUMN)

    class Base(BaseModel):
        model_config = ConfigDict(extra="forbid")

    model = create_request_model("SyntheticRequest", field_specs, base=Base)

    valid_payload = {
        "NUMERIC_FEATURE_1": 1.0,
        "NUMERIC_FEATURE_2": 2.0,
        "NUMERIC_FEATURE_3": 3.0,
        "CATEGORICAL_FEATURE": "A",
    }
    instance = model.model_validate(valid_payload)
    assert instance.NUMERIC_FEATURE_1 == 1.0

    with pytest.raises(ValidationError):
        model.model_validate({**valid_payload, "campo_desconhecido": 1})

    with pytest.raises(ValidationError):
        model.model_validate({k: v for k, v in valid_payload.items() if k != "NUMERIC_FEATURE_1"})


def test_create_request_model_attaches_json_schema_example(synthetic_dataset: pd.DataFrame) -> None:
    field_specs = build_field_specs(synthetic_dataset, target_column=TARGET_COLUMN)
    example = build_example_row(synthetic_dataset, field_specs)

    model = create_request_model("SyntheticRequestWithExample", field_specs, base=InspecaoMedidorBase, example=example)

    schema = model.model_json_schema()
    assert schema.get("example") == example
    assert schema.get("additionalProperties") is False


# ---------------------------------------------------------------------------
# Contratos reais da API (`LaudoCompletoRequest`/`LaudoSinteticoRequest`) —
# classes Pydantic FIXAS (sem geração dinâmica em runtime).
# ---------------------------------------------------------------------------


def test_inspecao_medidor_base_forbids_extra_fields() -> None:
    assert InspecaoMedidorBase.model_config.get("extra") == "forbid"


def test_laudo_sintetico_request_matches_retained_feature_columns() -> None:
    assert tuple(LaudoSinteticoRequest.model_fields) == RETAINED_FEATURE_COLUMNS
    for removed_feature in MANUALLY_REMOVED_FEATURES:
        assert removed_feature not in LaudoSinteticoRequest.model_fields


def test_laudo_completo_request_has_more_fields_than_sintetico() -> None:
    assert len(LaudoCompletoRequest.model_fields) > len(LaudoSinteticoRequest.model_fields)
    assert set(LaudoSinteticoRequest.model_fields).issubset(set(LaudoCompletoRequest.model_fields))


def test_laudo_completo_request_json_schema_includes_example() -> None:
    schema = LaudoCompletoRequest.model_json_schema()
    assert "example" in schema
    assert isinstance(schema["example"], dict)
    assert schema.get("additionalProperties") is False


def test_laudo_sintetico_request_json_schema_includes_example() -> None:
    schema = LaudoSinteticoRequest.model_json_schema()
    assert "example" in schema


def test_laudo_completo_request_accepts_its_own_example_payload() -> None:
    example = LaudoCompletoRequest.model_config["json_schema_extra"]["example"]
    instance = LaudoCompletoRequest.model_validate(example)
    assert instance is not None


def test_laudo_sintetico_request_accepts_its_own_example_payload() -> None:
    example = LaudoSinteticoRequest.model_config["json_schema_extra"]["example"]
    instance = LaudoSinteticoRequest.model_validate(example)
    assert instance is not None


def test_laudo_completo_request_rejects_unknown_field() -> None:
    example = dict(LaudoCompletoRequest.model_config["json_schema_extra"]["example"])
    example["CAMPO_QUE_NAO_EXISTE"] = "x"
    with pytest.raises(ValidationError):
        LaudoCompletoRequest.model_validate(example)


def test_laudo_completo_request_rejects_missing_required_field() -> None:
    example = dict(LaudoCompletoRequest.model_config["json_schema_extra"]["example"])
    required_field = next(
        name for name, field in LaudoCompletoRequest.model_fields.items() if field.is_required()
    )
    del example[required_field]
    with pytest.raises(ValidationError):
        LaudoCompletoRequest.model_validate(example)


def test_validate_laudo_completo_row_matches_laudo_completo_request() -> None:
    example = LaudoCompletoRequest.model_config["json_schema_extra"]["example"]
    validated = validate_laudo_completo_row(example)
    assert isinstance(validated, LaudoCompletoRequest)
