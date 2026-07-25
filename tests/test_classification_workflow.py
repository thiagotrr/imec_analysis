from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from machine_learning.classification import (
    ClassificationConfig,
    run_classification_workflow,
)
from machine_learning.classification.models import DEFAULT_CLASSIFIER_ORDER
from machine_learning.data_preparation import prepare_training_dataset
from machine_learning.feature_engineering import MIN_CLASS_PERCENTAGE_THRESHOLD

from conftest import TARGET_COLUMN


@pytest.fixture()
def prepared_dataset_dir(synthetic_dataset: pd.DataFrame, tmp_path: Path) -> Path:
    prepare_training_dataset(
        data_frame=synthetic_dataset,
        target_column=TARGET_COLUMN,
        output_dir=tmp_path,
        enable_pca=False,
        persist_artifacts=True,
        enable_exploration=False,
        min_class_percentage=15.0,
    )
    return tmp_path


def test_default_classifier_order_is_catboost_and_xgboost_only() -> None:
    assert DEFAULT_CLASSIFIER_ORDER == ("catboost", "xgboost")


def test_run_classification_workflow_end_to_end_without_resampling(prepared_dataset_dir: Path) -> None:
    """Smoke test de ponta a ponta com dados SINTÉTICOS (não reais).

    Valida que o pipeline de classificação executa sem erros com o novo
    critério de expurgo por percentual e produz métricas para os dois
    algoritmos padrão. As métricas aqui NÃO são representativas do problema
    real (dataset é sintético) — servem apenas como validação de execução do
    código (ver docs/task05_evolucao_pipeline_modelos.md).
    """
    config = ClassificationConfig(target_column=TARGET_COLUMN, output_dir=prepared_dataset_dir)
    result = run_classification_workflow(config=config)

    algorithms_run = {r.name for r in result.results}
    assert algorithms_run == {"catboost", "xgboost"}
    assert result.split_metadata["training_class_count"] == 3
    for classification_result in result.results:
        assert 0.0 <= classification_result.metrics.accuracy <= 1.0
        assert 0.0 <= classification_result.metrics.f1 <= 1.0

    assert result.artifacts is not None
    assert result.artifacts.summary_path.exists()
    assert result.artifacts.details_path.exists()
    with result.artifacts.details_path.open(encoding="utf-8") as file_obj:
        details = json.load(file_obj)
    # A preparação (fixture `prepared_dataset_dir`) já purgou o dataset a 15%
    # (camada "A"); a etapa de classificação aplica seu próprio expurgo por
    # cima usando o valor padrão atual (alinhado à camada "C"), que aqui não
    # remove nada adicional pois as 3 classes remanescentes já dominam o
    # dataset preparado.
    assert details["split"]["min_class_percentage"] == pytest.approx(MIN_CLASS_PERCENTAGE_THRESHOLD)
    assert details["split"]["training_class_count"] == 3


def test_run_classification_workflow_with_resampling_scenarios(prepared_dataset_dir: Path) -> None:
    config = ClassificationConfig(
        target_column=TARGET_COLUMN,
        output_dir=prepared_dataset_dir,
        algorithm_order=("catboost",),
        resampling_strategies=(None, "smote", "adasyn"),
    )
    result = run_classification_workflow(config=config)

    scenarios_run = {r.resampling for r in result.results}
    assert scenarios_run == {None, "smote", "adasyn"}
    assert len(result.results) == 3


def test_run_classification_workflow_with_cross_validation(prepared_dataset_dir: Path) -> None:
    config = ClassificationConfig(
        target_column=TARGET_COLUMN,
        output_dir=prepared_dataset_dir,
        algorithm_order=("xgboost",),
        enable_cross_validation=True,
        cross_validation_folds=3,
    )
    result = run_classification_workflow(config=config)

    assert len(result.results) == 1
    cross_validation_summary = result.results[0].cross_validation
    assert cross_validation_summary is not None
    for metric_name in ("f1_macro", "recall_macro", "roc_auc_macro"):
        assert cross_validation_summary[metric_name]["mean"] is not None
        assert len(cross_validation_summary[metric_name]["folds"]) == cross_validation_summary["cv_folds_used"]


def test_run_classification_workflow_with_hyperparameter_search(prepared_dataset_dir: Path) -> None:
    config = ClassificationConfig(
        target_column=TARGET_COLUMN,
        output_dir=prepared_dataset_dir,
        algorithm_order=("xgboost",),
        enable_hyperparameter_search=True,
        hyperparameter_search_iterations=2,
        hyperparameter_search_cv_folds=2,
    )
    result = run_classification_workflow(config=config)

    assert len(result.hyperparameter_search_results) == 1
    search_result = result.hyperparameter_search_results[0]
    assert search_result.algorithm == "xgboost"
    assert search_result.best_params
    assert result.results[0].hyperparameter_search is not None
