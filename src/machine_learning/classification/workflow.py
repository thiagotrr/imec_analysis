from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..data_preparation import get_model_dir
from ..feature_engineering import DEFAULT_TARGET_COLUMN
from .metrics import ClassificationMetrics, compute_classification_metrics, format_classification_metrics
from .models import DEFAULT_CLASSIFIER_ORDER, build_classifier_registry, run_classifier_training


DEFAULT_CLASSIFICATION_DIRNAME = "classification"
DEFAULT_CLASSIFICATION_SUMMARY_FILENAME = "classification_summary.csv"
DEFAULT_CLASSIFICATION_DETAILS_FILENAME = "classification_details.json"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_CLASS_COUNT = 2


@dataclass(frozen=True)
class ClassificationArtifacts:
    output_dir: Path
    summary_path: Path
    details_path: Path


@dataclass(frozen=True)
class ClassificationConfig:
    target_column: str = DEFAULT_TARGET_COLUMN
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = 42
    algorithm_order: tuple[str, ...] = DEFAULT_CLASSIFIER_ORDER
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT
    persist_artifacts: bool = True
    output_dir: str | Path | None = None
    dataset_path: str | Path | None = None


@dataclass(frozen=True)
class ClassificationResult:
    name: str
    metrics: ClassificationMetrics
    model_name: str


@dataclass(frozen=True)
class ClassificationWorkflowResult:
    results: list[ClassificationResult]
    consolidated: pd.DataFrame
    artifacts: ClassificationArtifacts | None
    split_metadata: dict[str, object]


def _resolve_prepared_dataset_path(output_dir: str | Path | None = None) -> Path:
    model_dir = get_model_dir(output_dir)
    return model_dir / "prepared_training_dataset.csv"


def _build_output_dir(output_dir: str | Path | None = None) -> Path:
    model_dir = get_model_dir(output_dir)
    classification_dir = model_dir / DEFAULT_CLASSIFICATION_DIRNAME
    classification_dir.mkdir(parents=True, exist_ok=True)
    return classification_dir


def _load_prepared_dataset(dataset_path: str | Path | None = None, output_dir: str | Path | None = None) -> pd.DataFrame:
    resolved_path = Path(dataset_path) if dataset_path is not None else _resolve_prepared_dataset_path(output_dir)
    return pd.read_csv(resolved_path)


def _split_dataset(
    data_frame: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    min_class_count: int,
):
    if target_column not in data_frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in prepared dataset.")
    class_counts = data_frame[target_column].value_counts().sort_index()
    rare_class_counts = class_counts[class_counts < min_class_count]
    filtered_frame = data_frame[~data_frame[target_column].isin(rare_class_counts.index)].copy()
    if filtered_frame.empty:
        raise ValueError("No target classes have enough samples for a stratified train/test split.")

    x = filtered_frame.drop(columns=[target_column])
    target_encoder = LabelEncoder()
    y = pd.Series(
        target_encoder.fit_transform(filtered_frame[target_column]),
        index=filtered_frame.index,
        name=target_column,
    )
    split = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    metadata = {
        "original_rows": int(len(data_frame)),
        "training_rows": int(len(filtered_frame)),
        "removed_rows": int(len(data_frame) - len(filtered_frame)),
        "original_class_count": int(class_counts.shape[0]),
        "training_class_count": int(y.nunique()),
        "min_class_count": int(min_class_count),
        "removed_rare_classes": {str(label): int(count) for label, count in rare_class_counts.items()},
        "target_label_mapping": {
            str(label): int(encoded_label)
            for encoded_label, label in enumerate(target_encoder.classes_)
        },
    }
    return (*split, metadata)


def _print_result(result: ClassificationResult) -> None:
    metrics = result.metrics
    print(f"\n{result.name}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1: {metrics.f1:.4f}")
    print(f"ROC-AUC: {metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "ROC-AUC: n/a")
    print(f"PR-AUC: {metrics.pr_auc:.4f}" if metrics.pr_auc is not None else "PR-AUC: n/a")


def run_classification_workflow(
    config: ClassificationConfig | None = None,
    output_dir: str | Path | None = None,
) -> ClassificationWorkflowResult:
    resolved_config = config or ClassificationConfig(output_dir=output_dir)
    prepared_dataset = _load_prepared_dataset(resolved_config.dataset_path, resolved_config.output_dir)
    x_train, x_test, y_train, y_test, split_metadata = _split_dataset(
        prepared_dataset,
        resolved_config.target_column,
        resolved_config.test_size,
        resolved_config.random_state,
        resolved_config.min_class_count,
    )
    n_classes = int(pd.Series(y_train).nunique())
    if split_metadata["removed_rows"]:
        print(
            "Classes raras removidas do treino/validacao: "
            f"{split_metadata['removed_rows']} linhas em {len(split_metadata['removed_rare_classes'])} classes."
        )

    registry = build_classifier_registry(random_state=resolved_config.random_state, n_classes=n_classes)
    results: list[ClassificationResult] = []
    for algorithm_name in resolved_config.algorithm_order:
        model = registry.get(algorithm_name)
        if model is None:
            continue
        training_result = run_classifier_training(algorithm_name, model, x_train, y_train, x_test)
        metrics = compute_classification_metrics(
            y_test,
            training_result.y_pred,
            training_result.y_score,
            classes=training_result.classes,
        )
        result = ClassificationResult(name=algorithm_name, metrics=metrics, model_name=model.__class__.__name__)
        results.append(result)
        _print_result(result)

    consolidated = pd.DataFrame([
        {
            "algorithm": result.name,
            "model": result.model_name,
            **format_classification_metrics(result.metrics),
        }
        for result in results
    ])

    artifacts = None
    if resolved_config.persist_artifacts:
        classification_dir = _build_output_dir(resolved_config.output_dir)
        summary_path = classification_dir / DEFAULT_CLASSIFICATION_SUMMARY_FILENAME
        details_path = classification_dir / DEFAULT_CLASSIFICATION_DETAILS_FILENAME
        consolidated.to_csv(summary_path, index=False)
        with details_path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "target_column": resolved_config.target_column,
                    "test_size": resolved_config.test_size,
                    "random_state": resolved_config.random_state,
                    "split": split_metadata,
                    "results": consolidated.to_dict(orient="records"),
                },
                file_obj,
                indent=2,
                ensure_ascii=False,
            )
        artifacts = ClassificationArtifacts(output_dir=classification_dir, summary_path=summary_path, details_path=details_path)

    if not consolidated.empty:
        print("\nConsolidado final")
        print(consolidated.to_string(index=False))

    return ClassificationWorkflowResult(
        results=results,
        consolidated=consolidated,
        artifacts=artifacts,
        split_metadata=split_metadata,
    )
