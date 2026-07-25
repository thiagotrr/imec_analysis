from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

DEFAULT_SEARCH_SCORING = "f1_macro"
"""Métrica-alvo da busca de hiperparâmetros: F1 macro pondera igualmente todas
as classes (inclusive as minoritárias), diferente de accuracy/F1 "micro" que
seriam dominados pelas classes majoritárias. ``"recall_macro"`` é a outra
opção recomendada pelo enunciado quando o foco é reduzir falsos negativos nas
classes raras.
"""
DEFAULT_SEARCH_ITERATIONS = 15
DEFAULT_SEARCH_CV_FOLDS = 3

CATBOOST_PARAM_DISTRIBUTIONS: dict[str, list[object]] = {
    "depth": [4, 6, 8, 10],
    "n_estimators": [300, 500, 800, 1200],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 9.0],
}

XGBOOST_PARAM_DISTRIBUTIONS: dict[str, list[object]] = {
    "max_depth": [3, 4, 6, 8, 10],
    "n_estimators": [200, 400, 600, 900, 1200],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
}

HYPERPARAMETER_SEARCH_SUPPORTED_ALGORITHMS = ("catboost", "xgboost")

_PARAM_DISTRIBUTIONS_BY_ALGORITHM: dict[str, dict[str, list[object]]] = {
    "catboost": CATBOOST_PARAM_DISTRIBUTIONS,
    "xgboost": XGBOOST_PARAM_DISTRIBUTIONS,
}


@dataclass(frozen=True)
class HyperparameterSearchResult:
    algorithm: str
    best_params: dict[str, object]
    best_score: float
    scoring: str
    cv_folds: int
    n_iter: int
    param_distributions: dict[str, list[object]] = field(default_factory=dict)


def get_param_distributions(algorithm: str) -> dict[str, list[object]] | None:
    return _PARAM_DISTRIBUTIONS_BY_ALGORITHM.get(algorithm)


def run_hyperparameter_search(
    algorithm: str,
    estimator: BaseEstimator,
    x_train,
    y_train,
    scoring: str = DEFAULT_SEARCH_SCORING,
    n_iter: int = DEFAULT_SEARCH_ITERATIONS,
    cv_folds: int = DEFAULT_SEARCH_CV_FOLDS,
    random_state: int = 42,
) -> tuple[BaseEstimator, HyperparameterSearchResult]:
    """Executa ``RandomizedSearchCV`` (poucas iterações) para ``algorithm`` otimizando ``scoring``.

    Retorna o melhor estimador já treinado (``refit=True``) em ``x_train``/``y_train``
    e um ``HyperparameterSearchResult`` com os hiperparâmetros escolhidos, o
    score obtido e o racional (espaço de busca utilizado), para documentação.
    """
    param_distributions = get_param_distributions(algorithm)
    if not param_distributions:
        raise ValueError(
            f"Sem espaço de busca de hiperparâmetros definido para o algoritmo '{algorithm}'. "
            f"Algoritmos suportados: {HYPERPARAMETER_SEARCH_SUPPORTED_ALGORITHMS}."
        )

    class_counts = np.bincount(np.asarray(y_train))
    resolved_cv_folds = max(2, min(cv_folds, int(np.min(class_counts[class_counts > 0]))))
    cross_validator = StratifiedKFold(n_splits=resolved_cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cross_validator,
        random_state=random_state,
        # n_jobs=1: ver racional em cross_validation.run_stratified_cross_validation
        # (evitar paralelismo aninhado processo x thread; CatBoost/XGBoost já usam
        # threads internamente para o treino de cada fit individual).
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    result = HyperparameterSearchResult(
        algorithm=algorithm,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        scoring=scoring,
        cv_folds=resolved_cv_folds,
        n_iter=n_iter,
        param_distributions=param_distributions,
    )
    return search.best_estimator_, result
