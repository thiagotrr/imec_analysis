from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate

DEFAULT_CV_FOLDS = 5
DEFAULT_CV_SCORING: dict[str, str] = {
    "f1_macro": "f1_macro",
    "recall_macro": "recall_macro",
    "roc_auc_macro": "roc_auc_ovr",
}
"""Métricas reportadas por fold na validação cruzada estratificada.

``roc_auc_ovr`` é o scorer nativo do scikit-learn para ROC-AUC multiclasse
"one-vs-rest", equivalente em espírito ao ``average="macro"`` usado no
restante do projeto (ver ``classification/metrics.py``).
"""


def run_stratified_cross_validation(
    estimator: BaseEstimator,
    x,
    y,
    cv_folds: int = DEFAULT_CV_FOLDS,
    random_state: int = 42,
    scoring: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Executa ``cross_validate`` com ``StratifiedKFold`` e resume média/desvio/valores por fold.

    ``estimator`` pode ser um classificador puro ou um Pipeline (ex.: com
    resampling via ``resampling.wrap_with_resampling``) — nesse caso, o
    resampling é reaplicado a cada fold apenas sobre a porção de treino
    daquele fold, evitando vazamento de dados para a validação.
    """
    resolved_scoring = scoring or DEFAULT_CV_SCORING
    n_splits = min(cv_folds, int(np.min(np.bincount(np.asarray(y)))) if len(y) else cv_folds)
    n_splits = max(2, n_splits)
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        estimator,
        x,
        y,
        cv=stratified_kfold,
        scoring=resolved_scoring,
        n_jobs=-1,
        error_score="raise",
    )

    summary: dict[str, dict[str, object]] = {"cv_folds_used": n_splits}
    for metric_name in resolved_scoring:
        key = f"test_{metric_name}"
        values = np.asarray(cv_results.get(key, []), dtype=float)
        summary[metric_name] = {
            "mean": float(np.mean(values)) if values.size else None,
            "std": float(np.std(values)) if values.size else None,
            "folds": [float(value) for value in values],
        }
    return summary
