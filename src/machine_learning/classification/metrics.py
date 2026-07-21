from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    average: str
    multiclass: bool


def _safe_auc_scores(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray | None,
    classes: np.ndarray | None = None,
) -> tuple[float | None, float | None, bool]:
    labels = np.asarray(y_true)
    unique_classes = np.asarray(classes) if classes is not None and len(classes) else np.unique(labels)
    multiclass = len(unique_classes) > 2
    if y_score is None:
        return None, None, multiclass

    if multiclass:
        binarized = label_binarize(labels, classes=unique_classes)
        if y_score.shape[1] != len(unique_classes):
            return None, None, multiclass
        valid_columns = np.where((binarized.sum(axis=0) > 0) & (binarized.sum(axis=0) < binarized.shape[0]))[0]
        if len(valid_columns) < 2:
            return None, None, multiclass
        roc_auc = roc_auc_score(binarized[:, valid_columns], y_score[:, valid_columns], average="macro")
        pr_auc = average_precision_score(binarized[:, valid_columns], y_score[:, valid_columns], average="macro")
        return float(roc_auc), float(pr_auc), multiclass

    positive_class = unique_classes[-1]
    binary_labels = (labels == positive_class).astype(int)
    if binary_labels.min() == binary_labels.max():
        return None, None, multiclass
    if y_score.ndim > 1:
        positive_index = int(np.where(unique_classes == positive_class)[0][0])
        y_score = y_score[:, positive_index]
    roc_auc = roc_auc_score(binary_labels, y_score)
    pr_auc = average_precision_score(binary_labels, y_score)
    return float(roc_auc), float(pr_auc), multiclass


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    average: str | None = None,
    classes: np.ndarray | None = None,
) -> ClassificationMetrics:
    labels = np.asarray(y_true)
    multiclass = len(np.unique(labels)) > 2
    resolved_average = average or ("macro" if multiclass else "binary")
    precision = precision_score(labels, y_pred, average=resolved_average, zero_division=0)
    recall = recall_score(labels, y_pred, average=resolved_average, zero_division=0)
    f1 = f1_score(labels, y_pred, average=resolved_average, zero_division=0)
    accuracy = accuracy_score(labels, y_pred)
    roc_auc, pr_auc, multiclass = _safe_auc_scores(labels, y_score, classes=classes)
    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        average=resolved_average,
        multiclass=multiclass,
    )


def format_classification_metrics(metrics: ClassificationMetrics) -> dict[str, float | str | bool | None]:
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "average": metrics.average,
        "multiclass": metrics.multiclass,
    }
