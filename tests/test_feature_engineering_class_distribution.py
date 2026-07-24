from __future__ import annotations

import pandas as pd
import pytest

from machine_learning.feature_engineering import (
    MIN_CLASS_PERCENTAGE_THRESHOLD,
    compute_class_distribution,
    filter_classes_by_percentage,
    identify_low_representation_classes,
)

from conftest import TARGET_COLUMN


def test_compute_class_distribution_counts_and_percentages_are_consistent(synthetic_dataset: pd.DataFrame) -> None:
    distribution = compute_class_distribution(synthetic_dataset[TARGET_COLUMN])

    assert set(distribution.columns) == {"class", "count", "percentage"}
    assert distribution["count"].sum() == len(synthetic_dataset)
    assert pytest.approx(distribution["percentage"].sum(), rel=1e-3) == 100.0
    # Ordenado da classe mais para a menos frequente.
    assert list(distribution["count"]) == sorted(distribution["count"], reverse=True)


def test_identify_low_representation_classes_uses_default_threshold(synthetic_dataset: pd.DataFrame) -> None:
    low_representation = identify_low_representation_classes(synthetic_dataset[TARGET_COLUMN])
    distribution = compute_class_distribution(synthetic_dataset[TARGET_COLUMN])

    expected = set(distribution.loc[distribution["percentage"] < MIN_CLASS_PERCENTAGE_THRESHOLD, "class"])
    assert set(low_representation) == expected
    # No dataset sintético, as 3 classes dominantes somam 86% e todas as demais
    # (10 classes raras) devem cair abaixo do limiar de 15%.
    assert len(low_representation) == 10


def test_filter_classes_by_percentage_removes_only_low_representation_rows(synthetic_dataset: pd.DataFrame) -> None:
    filtered_frame, metadata = filter_classes_by_percentage(synthetic_dataset, TARGET_COLUMN, min_percentage=15.0)

    assert metadata["original_class_count"] == 13
    assert metadata["retained_class_count"] == 3
    assert metadata["removed_class_count"] == 10
    assert metadata["removed_rows"] == len(synthetic_dataset) - len(filtered_frame)
    assert metadata["retained_rows"] == len(filtered_frame)
    # As classes retidas devem, cada uma, representar >= 15% do total original.
    after_distribution = pd.DataFrame(metadata["after_distribution"])
    assert (after_distribution["percentage"] >= 15.0).all()
    # E nenhuma linha do dataframe filtrado pertence a uma classe removida.
    removed_labels = {row["class"] for row in metadata["removed_classes"]}
    assert not filtered_frame[TARGET_COLUMN].astype(str).isin(removed_labels).any()


def test_filter_classes_by_percentage_raises_for_missing_target_column(synthetic_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        filter_classes_by_percentage(synthetic_dataset, "COLUNA_INEXISTENTE", min_percentage=15.0)
