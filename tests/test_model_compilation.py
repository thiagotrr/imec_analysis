from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from machine_learning.classification import (
    ClassificationConfig,
    ClassificationMetrics,
    ClassificationResult,
    run_classification_workflow,
)
from machine_learning.classification.model_compilation import (
    CHAMPION_STEM,
    compile_classification_results,
    compile_model_artifact,
    get_compiled_model_dir,
)
from machine_learning.classification.models import build_classifier_registry, run_classifier_training
from machine_learning.data_preparation import prepare_training_dataset

from conftest import TARGET_COLUMN


REQUIRED_METADATA_KEYS = {
    "algorithm",
    "resampling",
    "trained_at",
    "feature_columns",
    "target_classes",
    "metrics",
    "tier_thresholds",
    "class_weight_registry_path",
}


def _fit_simple_xgboost(synthetic_dataset: pd.DataFrame):
    feature_columns = ["NUMERIC_FEATURE_1", "NUMERIC_FEATURE_2", "NUMERIC_FEATURE_3"]
    x = synthetic_dataset[feature_columns]
    # Reindexa o target para 0..n-1 (exigido pelo XGBoost multiclasse).
    y = synthetic_dataset[TARGET_COLUMN].astype("category").cat.codes
    registry = build_classifier_registry(random_state=42, n_classes=y.nunique())
    training_result = run_classifier_training("xgboost", registry["xgboost"], x, y, x)
    return training_result.model, x, feature_columns, sorted(y.unique().tolist())


def test_compile_model_artifact_creates_loadable_pickle_with_matching_predictions(
    synthetic_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """O `.pkl` gerado deve ser carregável e reproduzir a MESMA predição do
    modelo em memória (critério de aceite explícito do plano, §1.3)."""
    model, x, feature_columns, target_classes = _fit_simple_xgboost(synthetic_dataset)

    pickle_path = compile_model_artifact(
        trained_model=model,
        algorithm="xgboost",
        resampling="smote",
        feature_columns=feature_columns,
        target_classes=target_classes,
        metrics={"accuracy": 0.9, "f1": 0.75},
        tier_thresholds={"A": 15.0, "B": 1.0, "C": 0.1},
        class_weight_registry_path=None,
        output_dir=tmp_path,
    )

    assert pickle_path.exists()
    assert pickle_path.name == "xgboost_smote.pkl"
    assert pickle_path.parent == get_compiled_model_dir(tmp_path)

    with pickle_path.open("rb") as file_obj:
        loaded_model = pickle.load(file_obj)

    assert list(loaded_model.predict(x)) == list(model.predict(x))
    assert (loaded_model.predict_proba(x) == model.predict_proba(x)).all()

    metadata_path = pickle_path.with_suffix(".json")
    assert metadata_path.exists()
    with metadata_path.open(encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    assert REQUIRED_METADATA_KEYS.issubset(metadata.keys())
    assert metadata["algorithm"] == "xgboost"
    assert metadata["resampling"] == "smote"
    assert metadata["feature_columns"] == feature_columns
    assert metadata["metrics"]["f1"] == 0.75


def test_compile_model_artifact_uses_none_label_when_no_resampling(
    synthetic_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    model, x, feature_columns, target_classes = _fit_simple_xgboost(synthetic_dataset)

    pickle_path = compile_model_artifact(
        trained_model=model,
        algorithm="xgboost",
        resampling=None,
        feature_columns=feature_columns,
        target_classes=target_classes,
        metrics={"f1": 0.5},
        output_dir=tmp_path,
    )

    assert pickle_path.name == "xgboost_none.pkl"
    metadata = json.loads(pickle_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["resampling"] == "none"
    assert metadata["class_weight_registry_path"] is None
    assert metadata["tier_thresholds"] is None


def test_compile_classification_results_elects_champion_by_f1_macro(
    synthetic_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    model, x, feature_columns, target_classes = _fit_simple_xgboost(synthetic_dataset)

    def _make_metrics(f1: float) -> ClassificationMetrics:
        return ClassificationMetrics(
            accuracy=0.8, precision=0.7, recall=0.7, f1=f1, roc_auc=0.9, pr_auc=0.8, average="macro", multiclass=True
        )

    low_f1_result = ClassificationResult(
        name="xgboost", metrics=_make_metrics(0.40), model_name="XGBClassifier", resampling=None, trained_model=model
    )
    high_f1_result = ClassificationResult(
        name="xgboost", metrics=_make_metrics(0.90), model_name="XGBClassifier", resampling="smote", trained_model=model
    )

    summary = compile_classification_results(
        results=[low_f1_result, high_f1_result],
        feature_columns=feature_columns,
        target_classes=target_classes,
        output_dir=tmp_path,
    )

    assert len(summary.compiled) == 2
    assert summary.champion is not None
    assert summary.champion.resampling == "smote"
    assert summary.champion.metadata["metrics"]["f1"] == pytest.approx(0.90)
    assert summary.champion.pickle_path.name == f"{CHAMPION_STEM}.pkl"
    assert summary.champion.metadata["champion_of"] == {"algorithm": "xgboost", "resampling": "smote"}


def test_compile_classification_results_returns_empty_summary_without_trained_models() -> None:
    metrics = ClassificationMetrics(
        accuracy=0.8, precision=0.7, recall=0.7, f1=0.6, roc_auc=None, pr_auc=None, average="macro", multiclass=True
    )
    result_without_model = ClassificationResult(name="xgboost", metrics=metrics, model_name="XGBClassifier")

    summary = compile_classification_results(
        results=[result_without_model],
        feature_columns=["a"],
        target_classes=[0, 1],
    )

    assert summary.compiled == []
    assert summary.champion is None


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


def test_run_classification_workflow_compiles_default_xgboost_smote_champion(prepared_dataset_dir: Path) -> None:
    """Integração ponta a ponta (dados SINTÉTICOS): com a configuração padrão
    (XGBoost + SMOTE, ver Task 006), a compilação deve gerar exatamente uma
    combinação (`xgboost_smote.pkl`) que também é o campeão (`champion.pkl`),
    ambos carregáveis e reproduzindo a mesma predição do modelo treinado em
    memória (`ClassificationResult.trained_model`)."""
    config = ClassificationConfig(target_column=TARGET_COLUMN, output_dir=prepared_dataset_dir)
    result = run_classification_workflow(config=config)

    assert result.compiled_models is not None
    assert len(result.compiled_models.compiled) == 1
    compiled_artifact = result.compiled_models.compiled[0]
    assert compiled_artifact.algorithm == "xgboost"
    assert compiled_artifact.resampling == "smote"

    champion = result.compiled_models.champion
    assert champion is not None
    assert champion.algorithm == "xgboost"
    assert champion.metadata["champion_of"] == {"algorithm": "xgboost", "resampling": "smote"}

    prepared_dataset = pd.read_csv(prepared_dataset_dir / "prepared_training_dataset.csv")
    x_full = prepared_dataset.drop(columns=[TARGET_COLUMN])

    trained_model = result.results[0].trained_model
    assert trained_model is not None

    with champion.pickle_path.open("rb") as file_obj:
        loaded_champion = pickle.load(file_obj)

    assert list(loaded_champion.predict(x_full)) == list(trained_model.predict(x_full))

    metadata = json.loads(champion.metadata_path.read_text(encoding="utf-8"))
    assert REQUIRED_METADATA_KEYS.issubset(metadata.keys())
    assert metadata["feature_columns"] == x_full.columns.tolist()
