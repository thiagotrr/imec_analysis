"""Executa Task 05 v3 com dados reais: preparação (14 classes, camadas A-C) +
classificação com CatBoost/XGBoost, cenários de resampling (None/SMOTE/ADASYN)
e StratifiedKFold — usando os hiperparâmetros padrão da Task 05 (sem busca de
hiperparâmetros, que é custosa e não é o objetivo desta rodada de validação).

Diferente de `run_task05_v2.py`, aqui o objetivo específico é validar se o
data augmentation (SMOTE/ADASYN) melhora as métricas nas classes das camadas
B e C, agora que `MIN_CLASS_PERCENTAGE_THRESHOLD` inclui essas classes por
padrão (ver `feature_engineering.CLASS_TIER_THRESHOLDS`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from machine_learning import (  # noqa: E402
    ClassificationConfig,
    print_preparation_summary,
    run_classification_workflow,
    run_preparation_workflow,
)

MODEL_DIR = ROOT / "model"
DETAILS_PATH = MODEL_DIR / "classification" / "classification_details_v3_real.json"


def main() -> None:
    print("=" * 60, flush=True)
    print("Task 05 v3 - preparacao com dados reais (camadas A-C, 14 classes)", flush=True)
    print("=" * 60, flush=True)
    prep = run_preparation_workflow(open_browser=False, enable_exploration=False)
    print_preparation_summary(prep.preparation_result)

    print("\n" + "=" * 60, flush=True)
    print("Task 05 v3 - CatBoost/XGBoost + SMOTE/ADASYN + StratifiedKFold (k=5)", flush=True)
    print("=" * 60, flush=True)
    config = ClassificationConfig(
        # A Task 006 mudou o PADRÃO de `algorithm_order`/`resampling_strategies`
        # para XGBoost+SMOTE apenas (ver `classification.models.DEFAULT_CLASSIFIER_ORDER`
        # e `ClassificationConfig.resampling_strategies`); este script documenta
        # explicitamente o escopo histórico da v3 (comparar CatBoost x XGBoost
        # nos 3 cenários de resampling), por isso informa os dois parâmetros
        # explicitamente em vez de depender dos novos defaults.
        algorithm_order=("catboost", "xgboost"),
        enable_hyperparameter_search=False,
        resampling_strategies=(None, "smote", "adasyn"),
        enable_cross_validation=False,
        persist_artifacts=True,
        # A compilação de modelos (`.pkl` em `model/compiled/`) é uma etapa nova
        # da Task 006, fora do escopo desta rodada comparativa da v3.
        compile_artifacts=False,
    )
    result = run_classification_workflow(config=config)

    if result.artifacts is not None:
        DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DETAILS_PATH.write_text(
            result.artifacts.details_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"\nDetalhes: {DETAILS_PATH}", flush=True)

    rows = []
    for r in result.results:
        rows.append(
            {
                "algorithm": r.name,
                "resampling": r.resampling or "none",
                "accuracy": r.metrics.accuracy,
                "precision": r.metrics.precision,
                "recall": r.metrics.recall,
                "f1": r.metrics.f1,
                "roc_auc": r.metrics.roc_auc,
                "per_class": r.per_class,
            }
        )
    summary_path = MODEL_DIR / "classification" / "task05_v3_run_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumo da execucao: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
