from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


DEFAULT_CLASSIFIER_ORDER = ("catboost", "xgboost", "knn", "svm")


@dataclass(frozen=True)
class ClassifierTrainingResult:
    name: str
    model: BaseEstimator
    y_pred: np.ndarray
    y_score: np.ndarray | None
    classes: np.ndarray | None


def build_classifier_registry(random_state: int = 42, n_classes: int = 2) -> dict[str, BaseEstimator]:
    registry: dict[str, BaseEstimator] = {
        "knn": KNeighborsClassifier(n_neighbors=7),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state),
    }
    if CatBoostClassifier is not None:
        registry["catboost"] = CatBoostClassifier(
            verbose=False,
            random_seed=random_state,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
        )
    if XGBClassifier is not None:
        xgb_params: dict[str, object] = {
            "random_state": random_state,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        }
        if n_classes > 2:
            xgb_params.update({"objective": "multi:softprob", "num_class": n_classes})
        else:
            xgb_params.update({"objective": "binary:logistic"})
        registry["xgboost"] = XGBClassifier(**xgb_params)
    return registry


def run_classifier_training(name: str, model: BaseEstimator, x_train, y_train, x_test) -> ClassifierTrainingResult:
    trained_model = model.fit(x_train, y_train)
    y_pred = np.asarray(trained_model.predict(x_test)).ravel()
    y_score = trained_model.predict_proba(x_test) if hasattr(trained_model, "predict_proba") else None
    classes = np.asarray(getattr(trained_model, "classes_", [])) if hasattr(trained_model, "classes_") else None
    return ClassifierTrainingResult(name=name, model=trained_model, y_pred=y_pred, y_score=y_score, classes=classes)
