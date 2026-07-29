"""Gera o cache de especificação de campos + JSON Schemas dos contratos de laudo.

NOTA (revisão Task 006): os contratos da API em
`src/api/models/inspecao_request.py` são agora classes Pydantic **FIXAS**
(não reconstruídas em runtime). Este script permanece apenas como utilitário
offline para inspecionar/atualizar artefatos em `model/schemas/` quando o
dataset de origem mudar — a API NÃO depende mais desse cache.

Artefatos gerados:
- `model/schemas/raw_dataset_field_spec.json`
- `model/schemas/laudo_completo_schema.json` / `laudo_sintetico_schema.json`

Uso:
    python scripts/generate_pydantic_schema.py
    python scripts/generate_pydantic_schema.py --dataset-path outro_laudo.xlsx --output-dir model/schemas
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.models.inspecao_request import InspecaoMedidorBase  # noqa: E402
from api.schema_generation import (  # noqa: E402
    build_example_row,
    build_field_specs,
    build_laudo_sintetico_field_specs,
    create_request_model,
    dump_field_specs,
)
from machine_learning.feature_engineering import (  # noqa: E402
    DEFAULT_TARGET_COLUMN,
    load_dataset,
    resolve_dataset_path,
)

DEFAULT_SCHEMAS_DIRNAME = "schemas"
DEFAULT_FIELD_SPEC_FILENAME = "raw_dataset_field_spec.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", default=None, help="Caminho do dataset real (padrão: resultado_laudo_afericao.xlsx).")
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN, help=f"Coluna target a excluir dos contratos (padrão: {DEFAULT_TARGET_COLUMN}).")
    parser.add_argument("--output-dir", default=None, help="Diretório de saída (padrão: model/schemas).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    dataset_path = resolve_dataset_path(args.dataset_path)
    print(f"Lendo dataset real: {dataset_path}")
    data_frame = load_dataset(dataset_path)
    print(f"Dataset lido: {data_frame.shape[0]} linhas x {data_frame.shape[1]} colunas.")

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "model" / DEFAULT_SCHEMAS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    full_field_specs = build_field_specs(data_frame, target_column=args.target_column)
    sintetico_field_specs = build_laudo_sintetico_field_specs(data_frame, target_column=args.target_column)
    example_row = build_example_row(data_frame, full_field_specs)

    cache_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_path": str(dataset_path),
        "target_column": args.target_column,
        "field_specs": dump_field_specs(full_field_specs),
        "example_row": example_row,
    }
    field_spec_path = output_dir / DEFAULT_FIELD_SPEC_FILENAME
    field_spec_path.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Cache de especificação de campos: {field_spec_path} ({len(full_field_specs)} colunas)")

    laudo_completo_model = create_request_model(
        "LaudoCompletoRequest", full_field_specs, base=InspecaoMedidorBase, example=example_row
    )
    sintetico_example = {key: value for key, value in example_row.items() if key in sintetico_field_specs}
    laudo_sintetico_model = create_request_model(
        "LaudoSinteticoRequest", sintetico_field_specs, base=InspecaoMedidorBase, example=sintetico_example
    )

    for name, model in (("laudo_completo", laudo_completo_model), ("laudo_sintetico", laudo_sintetico_model)):
        schema_path = output_dir / f"{name}_schema.json"
        schema_path.write_text(
            json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON Schema ({name}): {schema_path} ({len(model.model_fields)} campos)")


if __name__ == "__main__":
    main()
