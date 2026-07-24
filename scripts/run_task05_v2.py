"""Executa pipeline Task 05 v2 com dados reais e persiste artefatos de avaliação."""
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
PHASE1_DETAILS = MODEL_DIR / "classification" / "classification_details_phase1.json"
PHASE2_DETAILS = MODEL_DIR / "classification" / "classification_details_phase2.json"

LOW_METRIC_THRESHOLD = 0.70


def _macro_metrics(results) -> dict[str, float]:
    """Retorna pior recall/F1/ROC-AUC macro entre os resultados sem resampling."""
    candidates = [
        r for r in results
        if (r.resampling or "none") in (None, "none")
    ]
    if not candidates:
        candidates = results
    recalls = [r.metrics.recall for r in candidates]
    f1s = [r.metrics.f1 for r in candidates]
    roc_aucs = [r.metrics.roc_auc for r in candidates if r.metrics.roc_auc is not None]
    return {
        "recall_min": min(recalls) if recalls else 0.0,
        "f1_min": min(f1s) if f1s else 0.0,
        "roc_auc_min": min(roc_aucs) if roc_aucs else 0.0,
    }


def _needs_phase2(results) -> bool:
    metrics = _macro_metrics(results)
    return any(
        metrics[key] < LOW_METRIC_THRESHOLD
        for key in ("recall_min", "f1_min", "roc_auc_min")
    )


def main() -> None:
    print("=" * 60)
    print("Task 05 v2 — preparação com dados reais")
    print("=" * 60)
    prep = run_preparation_workflow(open_browser=False, enable_exploration=True)
    print_preparation_summary(prep.preparation_result)

    print("\n" + "=" * 60)
    print("Task 05 v2 — Fase 1: CatBoost/XGBoost + busca de hiperparâmetros")
    print("=" * 60)
    phase1_config = ClassificationConfig(
        enable_hyperparameter_search=True,
        resampling_strategies=(None,),
        enable_cross_validation=False,
        persist_artifacts=True,
    )
    phase1 = run_classification_workflow(config=phase1_config)
    if phase1.artifacts is not None:
        phase1_path = PHASE1_DETAILS
        phase1_path.parent.mkdir(parents=True, exist_ok=True)
        phase1_path.write_text(
            phase1.artifacts.details_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"Detalhes Fase 1: {phase1_path}")

    run_phase2 = _needs_phase2(phase1.results)
    macro = _macro_metrics(phase1.results)
    print(
        f"\nCritério Fase 2 (limiar={LOW_METRIC_THRESHOLD}): "
        f"recall_min={macro['recall_min']:.4f}, f1_min={macro['f1_min']:.4f}, "
        f"roc_auc_min={macro['roc_auc_min']:.4f} -> "
        f"{'executar Fase 2' if run_phase2 else 'métricas OK, Fase 2 opcional'}"
    )

    phase2 = None
    if run_phase2:
        print("\n" + "=" * 60)
        print("Task 05 v2 — Fase 2: SMOTE/ADASYN + StratifiedKFold")
        print("=" * 60)
        phase2_config = ClassificationConfig(
            enable_hyperparameter_search=False,
            resampling_strategies=(None, "smote", "adasyn"),
            enable_cross_validation=True,
            cross_validation_folds=5,
            persist_artifacts=True,
        )
        phase2 = run_classification_workflow(config=phase2_config)
        if phase2.artifacts is not None:
            phase2_path = PHASE2_DETAILS
            phase2_path.write_text(
                phase2.artifacts.details_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            print(f"Detalhes Fase 2: {phase2_path}")

    summary = {
        "phase1_macro_min": macro,
        "phase2_executed": run_phase2,
        "phase1_results_count": len(phase1.results),
        "phase2_results_count": len(phase2.results) if phase2 else 0,
    }
    summary_path = MODEL_DIR / "classification" / "task05_v2_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumo da execução: {summary_path}")


if __name__ == "__main__":
    main()
