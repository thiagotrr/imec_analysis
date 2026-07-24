from __future__ import annotations

from pathlib import Path

import pandas as pd

from machine_learning.classification import ClassificationConfig, run_classification_workflow
from machine_learning.data_preparation import run_preparation_workflow

from conftest import TARGET_COLUMN


def test_full_pipeline_with_default_pca_resampling_cv_and_search(
    synthetic_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """Smoke test de ponta a ponta usando a configuração "de produção" (PCA
    habilitado por padrão em `prepare_training_dataset`), combinando todas as
    features novas: expurgo por percentual, resampling (SMOTE/ADASYN),
    StratifiedKFold e busca de hiperparâmetros — tudo com dados SINTÉTICOS.

    Isso valida que o caminho de código mais próximo do uso real (`src/main.py`)
    funciona sem erros; as métricas seguem sem valor de negócio real (ver
    docs/task05_evolucao_pipeline_modelos.md).
    """
    # min_class_percentage=15.0 (camada "A") fixado explicitamente: este teste
    # valida a integração PCA + resampling + CV + busca de hiperparâmetros,
    # não o sistema de camadas em si (ver test_feature_engineering_class_distribution.py).
    # Com o valor padrão atual (camada "C", ~0.1%) o dataset sintético retém
    # classes rarissimas de propósito (para simular a cauda real) com poucas
    # amostras por fold de CV — cenário em que SMOTE/ADASYN falham por design
    # (n_neighbors > amostras disponíveis), não por bug: é exatamente a
    # limitação de confiabilidade documentada em
    # docs/task05_analise_data_augmentation.md para classes de baixo volume.
    run_preparation_workflow(
        data_frame=synthetic_dataset,
        target_column=TARGET_COLUMN,
        output_dir=tmp_path,
        open_browser=False,
        enable_exploration=False,
        min_class_percentage=15.0,
    )

    config = ClassificationConfig(
        target_column=TARGET_COLUMN,
        output_dir=tmp_path,
        min_class_percentage=15.0,
        resampling_strategies=(None, "smote", "adasyn"),
        enable_cross_validation=True,
        cross_validation_folds=3,
        enable_hyperparameter_search=True,
        hyperparameter_search_iterations=2,
        hyperparameter_search_cv_folds=2,
    )
    result = run_classification_workflow(config=config)

    assert len(result.results) == 2 * 3  # 2 algoritmos x 3 cenários de resampling
    assert len(result.hyperparameter_search_results) == 2  # catboost + xgboost
    for classification_result in result.results:
        assert classification_result.cross_validation is not None
        assert 0.0 <= classification_result.metrics.accuracy <= 1.0
    assert result.artifacts is not None and result.artifacts.details_path.exists()
