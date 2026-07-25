"""Pipeline lite com dados reais (pandas + catboost + xgboost, sem scipy/sklearn).

Usado quando scipy/sklearn estão indisponíveis por política de segurança do SO.
Aplica o mesmo expurgo de 15% e hiperparâmetros propostos na Task 05.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
FE_DIR = ROOT / "src" / "machine_learning"
sys.path.insert(0, str(FE_DIR))

import feature_engineering as fe  # noqa: E402

TARGET = fe.DEFAULT_TARGET_COLUMN
THRESHOLD = fe.MIN_CLASS_PERCENTAGE_THRESHOLD
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

CATBOOST_PARAMS = {
    "depth": 6,
    "iterations": 600,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "random_seed": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
}


def _load_clean_frame() -> pd.DataFrame:
    path = FE_DIR / fe.DEFAULT_DATASET_FILENAME
    if not path.exists():
        path = ROOT / fe.DEFAULT_DATASET_FILENAME
    frame = pd.read_excel(path)
    frame = frame.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    filtered, _ = fe.filter_classes_by_percentage(frame, TARGET, min_percentage=THRESHOLD)
    return filtered


def _prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [
        c for c in fe.get_manually_removed_features(frame, TARGET)
        if c in frame.columns
    ]
    features = frame.drop(columns=drop_cols + [TARGET], errors="ignore").copy()
    target = frame[TARGET].astype(str)

    for col in features.select_dtypes(include=["object", "string", "category"]).columns:
        features[col] = features[col].astype(str).fillna("__missing__")
        features[col], _ = pd.factorize(features[col])

    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            features[col] = features[col].fillna(features[col].median())
        else:
            features[col] = pd.factorize(features[col].astype(str))[0]

    features = features.loc[:, features.nunique(dropna=False) > 1]
    return features, target


def _stratified_split(x: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in y.unique():
        label_indices = np.where(y.values == label)[0]
        rng.shuffle(label_indices)
        n_test = max(1, int(round(len(label_indices) * test_size)))
        test_idx.extend(label_indices[:n_test].tolist())
        train_idx.extend(label_indices[n_test:].tolist())
    return x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _encode_target(y: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    classes = sorted(y.unique())
    mapping = {label: idx for idx, label in enumerate(classes)}
    encoded = np.array([mapping[label] for label in y], dtype=int)
    return encoded, mapping


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    return np.array([total / (n_classes * counts[label]) for label in y], dtype=float)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[true_label, pred_label] += 1
    return matrix


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float | None]:
    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    matrix = _confusion_matrix(y_true, y_pred, n_classes)
    accuracy = float((y_true == y_pred).mean())

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in range(n_classes):
        tp = matrix[cls, cls]
        fp = matrix[:, cls].sum() - tp
        fn = matrix[cls, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    roc_auc = None
    if y_score is not None and n_classes == 2:
        positive = (y_true == 1).astype(int)
        order = np.argsort(-y_score)
        scores = y_score[order]
        labels = positive[order]
        tps = np.cumsum(labels)
        fps = np.cumsum(1 - labels)
        tpr = tps / max(labels.sum(), 1)
        fpr = fps / max((1 - labels).sum(), 1)
        roc_auc = float(np.trapezoid(tpr, fpr))

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "roc_auc": roc_auc,
        "pr_auc": None,
        "average": "macro",
        "multiclass": n_classes > 2,
    }


def _random_oversample(x: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = Counter(y)
    max_count = max(counts.values())
    indices: list[int] = list(range(len(y)))
    for cls, count in counts.items():
        cls_indices = np.where(y == cls)[0]
        extra = max_count - count
        if extra > 0:
            indices.extend(rng.choice(cls_indices, size=extra, replace=True).tolist())
    rng.shuffle(indices)
    return x.iloc[indices].reset_index(drop=True), y[indices]


def _train_catboost(x_train, y_train, x_test) -> tuple[np.ndarray, np.ndarray]:
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).astype(int).ravel()
    y_score = model.predict_proba(x_test)[:, 1]
    return y_pred, y_score


def _train_xgboost(x_train, y_train, x_test, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    import xgboost as xgb

    params = {
        "max_depth": XGBOOST_PARAMS["max_depth"],
        "eta": XGBOOST_PARAMS["learning_rate"],
        "min_child_weight": XGBOOST_PARAMS["min_child_weight"],
        "subsample": XGBOOST_PARAMS["subsample"],
        "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": RANDOM_STATE,
    }
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weight)
    dtest = xgb.DMatrix(x_test)
    booster = xgb.train(params, dtrain, num_boost_round=int(XGBOOST_PARAMS["n_estimators"]))
    y_score = booster.predict(dtest)
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def _evaluate(algorithm: str, x_train, y_train, x_test, y_test, resampling: str | None) -> dict[str, object]:
    x_fit, y_fit = x_train, y_train
    if resampling == "random_oversample":
        x_fit, y_fit = _random_oversample(x_train, y_train, RANDOM_STATE)

    if algorithm == "catboost":
        y_pred, y_score = _train_catboost(x_fit, y_fit, x_test)
    else:
        weights = _balanced_weights(y_fit) if resampling is None else None
        y_pred, y_score = _train_xgboost(x_fit, y_fit, x_test, sample_weight=weights)

    metrics = _classification_metrics(y_test, y_pred, y_score)
    return {
        "algorithm": algorithm,
        "resampling": resampling or "none",
        **metrics,
    }


def _stratified_kfold_cv(algorithm: str, x: pd.DataFrame, y: np.ndarray, folds: int) -> dict[str, float]:
    fold_metrics: dict[str, list[float]] = {"f1": [], "recall": [], "roc_auc": []}
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}
    rng = np.random.default_rng(RANDOM_STATE)

    for fold in range(folds):
        test_idx: list[int] = []
        train_idx: list[int] = []
        for cls, idx in class_indices.items():
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            split = len(shuffled) // folds
            start = fold * split
            end = (fold + 1) * split if fold < folds - 1 else len(shuffled)
            test_idx.extend(shuffled[start:end].tolist())
            train_idx.extend(shuffled[:start].tolist())
            train_idx.extend(shuffled[end:].tolist())

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if algorithm == "catboost":
            y_pred, y_score = _train_catboost(x_train, y_train, x_test)
        else:
            y_pred, y_score = _train_xgboost(x_train, y_train, x_test, sample_weight=_balanced_weights(y_train))

        metrics = _classification_metrics(y_test, y_pred, y_score)
        fold_metrics["f1"].append(metrics["f1"])
        fold_metrics["recall"].append(metrics["recall"])
        if metrics["roc_auc"] is not None:
            fold_metrics["roc_auc"].append(metrics["roc_auc"])

    summary: dict[str, object] = {"cv_folds_used": folds}
    for key, values in fold_metrics.items():
        if values:
            summary[f"{key}_macro"] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def main() -> None:
    frame = _load_clean_frame()
    x, y_raw = _prepare_features(frame)
    y_full, label_map = _encode_target(y_raw)
    x_train, x_test, y_train, y_test = _stratified_split(
        x, pd.Series(y_full, index=x.index), TEST_SIZE, RANDOM_STATE
    )
    y_train = y_train.to_numpy(dtype=int)
    y_test = y_test.to_numpy(dtype=int)

    results: list[dict[str, object]] = []
    for algorithm in ("catboost", "xgboost"):
        for resampling in (None, "random_oversample"):
            result = _evaluate(algorithm, x_train, y_train, x_test, y_test, resampling)
            if resampling is None:
                result["cross_validation"] = _stratified_kfold_cv(algorithm, x, y_full, CV_FOLDS)
            results.append(result)
            print(
                f"{algorithm} resampling={result['resampling']}: "
                f"acc={result['accuracy']:.4f} f1={result['f1']:.4f} "
                f"recall={result['recall']:.4f} roc_auc={result['roc_auc']}"
            )

    output_dir = ROOT / "model" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "lite_no_sklearn",
        "note": (
            "Execução com dados reais via pipeline lite (sem pré-processamento sklearn completo). "
            "Expurgo de 15% e hiperparâmetros Task 05 aplicados. "
            "'random_oversample' substitui SMOTE/ADASYN neste modo por indisponibilidade do imbalanced-learn."
        ),
        "target_column": TARGET,
        "min_class_percentage": THRESHOLD,
        "retained_classes": list(label_map.keys()),
        "label_mapping": label_map,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "results": results,
    }
    out_path = output_dir / "classification_details_v2_real_lite.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo: {out_path}")


if __name__ == "__main__":
    main()
