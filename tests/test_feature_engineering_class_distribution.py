from __future__ import annotations

import pandas as pd
import pytest

from machine_learning.feature_engineering import (
    CLASS_TIER_DISCARD_LABEL,
    CLASS_TIER_THRESHOLDS,
    MIN_CLASS_PERCENTAGE_THRESHOLD,
    build_class_weight_registry,
    classify_class_tier,
    compute_balanced_class_weights,
    compute_class_distribution,
    filter_classes_by_percentage,
    identify_low_representation_classes,
    qualify_class_distribution,
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
    # Usa min_percentage=15.0 explicitamente (camada "A") para desacoplar este
    # teste do valor de MIN_CLASS_PERCENTAGE_THRESHOLD (hoje alinhado à camada
    # "C" = 0.1%, ver test_classify_class_tier_* abaixo para a cobertura do
    # sistema de camadas em si).
    low_representation = identify_low_representation_classes(synthetic_dataset[TARGET_COLUMN], min_percentage=15.0)
    distribution = compute_class_distribution(synthetic_dataset[TARGET_COLUMN])

    expected = set(distribution.loc[distribution["percentage"] < 15.0, "class"])
    assert set(low_representation) == expected
    # No dataset sintético, as 3 classes dominantes somam 86% e todas as demais
    # (10 classes raras) devem cair abaixo do limiar de 15%.
    assert len(low_representation) == 10


def test_min_class_percentage_threshold_is_aligned_with_tier_c_lower_bound() -> None:
    # O expurgo padrão deve descartar apenas a camada "D" (cauda estatística),
    # retendo A/B/C — ver feature_engineering.MIN_CLASS_PERCENTAGE_THRESHOLD.
    assert MIN_CLASS_PERCENTAGE_THRESHOLD == CLASS_TIER_THRESHOLDS["C"]


def test_classify_class_tier_uses_configured_thresholds() -> None:
    assert classify_class_tier(46.0) == "A"
    assert classify_class_tier(CLASS_TIER_THRESHOLDS["A"]) == "A"
    assert classify_class_tier(5.0) == "B"
    assert classify_class_tier(CLASS_TIER_THRESHOLDS["B"]) == "B"
    assert classify_class_tier(0.5) == "C"
    assert classify_class_tier(CLASS_TIER_THRESHOLDS["C"]) == "C"
    assert classify_class_tier(0.01) == CLASS_TIER_DISCARD_LABEL


def test_qualify_class_distribution_adds_tier_column_without_mutating_input(
    synthetic_dataset: pd.DataFrame,
) -> None:
    distribution = compute_class_distribution(synthetic_dataset[TARGET_COLUMN])
    qualified = qualify_class_distribution(distribution)

    assert "tier" not in distribution.columns
    assert set(qualified.columns) == {"class", "count", "percentage", "tier"}
    assert set(qualified["tier"]).issubset({"A", "B", "C", CLASS_TIER_DISCARD_LABEL})
    # As 3 classes dominantes (>= 15%) devem ser qualificadas na camada "A".
    dominant = qualified.loc[qualified["percentage"] >= 15.0, "tier"]
    assert (dominant == "A").all()


def test_compute_balanced_class_weights_penalizes_smaller_classes(synthetic_dataset: pd.DataFrame) -> None:
    distribution = compute_class_distribution(synthetic_dataset[TARGET_COLUMN])
    weighted = compute_balanced_class_weights(distribution)

    assert "weight" in weighted.columns
    # Classe com mais amostras deve ter peso menor que classe com menos amostras.
    largest = weighted.loc[weighted["count"].idxmax()]
    smallest = weighted.loc[weighted["count"].idxmin()]
    assert largest["weight"] < smallest["weight"]


def test_build_class_weight_registry_retains_only_classes_above_threshold(
    synthetic_dataset: pd.DataFrame,
) -> None:
    registry = build_class_weight_registry(synthetic_dataset, TARGET_COLUMN, min_percentage=15.0)

    assert registry["retained_class_count"] == 3
    assert registry["discarded_class_count"] == 10
    assert len(registry["classes"]) == 3
    for class_entry in registry["classes"]:
        assert class_entry["percentage"] >= 15.0
        assert class_entry["tier"] == "A"
        assert class_entry["weight"] > 0
    assert registry["retained_rows"] + registry["discarded_rows"] == len(synthetic_dataset)


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
