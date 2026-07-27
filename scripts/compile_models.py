"""Compila modelo(s) treinado(s) em artefatos `.pkl` + `.json` (Task 006, §1).

Diferente de `run_task05_v3.py` (que refaz a preparação completa do dataset
real — leitura do Excel, limpeza, `TruncatedSVD`, etc., a etapa mais lenta do
pipeline), este script roda SOMENTE a etapa de classificação
(`run_classification_workflow`) sobre o dataset JÁ preparado em
`model/prepared_training_dataset.csv`, e aciona a compilação
(`ClassificationConfig.compile_artifacts=True`, plugada ao final de
`run_classification_workflow` — ver `classification.model_compilation`).

Por padrão treina apenas XGBoost+SMOTE (novo cenário padrão da Task 006 — ver
`classification.models.DEFAULT_CLASSIFIER_ORDER` e
`ClassificationConfig.resampling_strategies`), mas aceita flags para incluir
CatBoost e/ou outros cenários de resampling (nenhum/ADASYN) quando desejado —
CatBoost/ADASYN permanecem implementados, apenas fora do fluxo padrão (ver
docs/task006_proximos_passos.md e docs/task05_evolucao_pipeline_modelos_v3.md).

Uso:
    python scripts/compile_models.py
    python scripts/compile_models.py --include-catboost
    python scripts/compile_models.py --algorithm xgboost --algorithm catboost
    python scripts/compile_models.py --resampling none smote adasyn
    python scripts/compile_models.py --dataset-path model/prepared_training_dataset.csv --output-dir model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from machine_learning.classification import ClassificationConfig, run_classification_workflow  # noqa: E402
from machine_learning.classification.models import DEFAULT_CLASSIFIER_ORDER  # noqa: E402


def _parse_resampling_token(value: str) -> str | None:
    normalized = value.strip().lower()
    return None if normalized in {"none", "nenhum", ""} else normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--include-catboost",
        action="store_true",
        help='Inclui CatBoost além do XGBoost padrão (equivalente a --algorithm catboost --algorithm xgboost).',
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        choices=["catboost", "xgboost"],
        help="Especifica algoritmo(s) explicitamente (repita a flag para múltiplos). Sobrepõe --include-catboost.",
    )
    parser.add_argument(
        "--resampling",
        nargs="+",
        default=None,
        metavar="{none,smote,adasyn}",
        help='Estratégias de resampling a comparar (ex.: "none smote adasyn"). Padrão: smote apenas.',
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Caminho do dataset já preparado (padrão: model/prepared_training_dataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de modelo/artefatos (padrão: model/).",
    )
    return parser


def resolve_algorithm_order(args: argparse.Namespace) -> tuple[str, ...]:
    if args.algorithms:
        # Preserva ordem sem duplicatas, respeitando a escolha explícita do usuário.
        return tuple(dict.fromkeys(args.algorithms))
    if args.include_catboost:
        return ("catboost", "xgboost")
    return DEFAULT_CLASSIFIER_ORDER


def resolve_resampling_strategies(args: argparse.Namespace) -> tuple[str | None, ...] | None:
    if args.resampling is None:
        return None  # usa o default de ClassificationConfig (("smote",))
    return tuple(_parse_resampling_token(value) for value in args.resampling)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    config_kwargs: dict[str, object] = {
        "algorithm_order": resolve_algorithm_order(args),
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "compile_artifacts": True,
    }
    resampling_strategies = resolve_resampling_strategies(args)
    if resampling_strategies is not None:
        config_kwargs["resampling_strategies"] = resampling_strategies

    print("=" * 60)
    print("Task 006 - Compilação de modelos")
    print(f"algorithm_order={config_kwargs['algorithm_order']}")
    print(f"resampling_strategies={resampling_strategies or '(default: smote)'}")
    print("=" * 60)

    config = ClassificationConfig(**config_kwargs)
    result = run_classification_workflow(config=config)

    if result.compiled_models is None or not result.compiled_models.compiled:
        print("\nNenhum modelo compilado (nenhum resultado com modelo treinado disponível).")
        return

    print(f"\n{len(result.compiled_models.compiled)} combinação(ões) compilada(s):")
    for artifact in result.compiled_models.compiled:
        f1 = artifact.metadata.get("metrics", {}).get("f1")
        f1_display = f"{f1:.4f}" if isinstance(f1, (int, float)) else "n/a"
        print(
            f"  - {artifact.algorithm} (resampling={artifact.resampling or 'none'}): "
            f"f1_macro={f1_display} -> {artifact.pickle_path.name}"
        )

    champion = result.compiled_models.champion
    if champion is not None:
        print(f"\nModelo campeão: {champion.algorithm} (resampling={champion.resampling or 'none'})")
        print(f"  pickle: {champion.pickle_path}")
        print(f"  metadata: {champion.metadata_path}")


if __name__ == "__main__":
    main()
