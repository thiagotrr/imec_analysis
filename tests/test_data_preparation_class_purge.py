from __future__ import annotations

from pathlib import Path

import pandas as pd

from machine_learning.data_preparation import prepare_training_dataset

from conftest import TARGET_COLUMN


def test_prepare_training_dataset_purges_low_representation_classes(
    synthetic_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    result = prepare_training_dataset(
        data_frame=synthetic_dataset,
        target_column=TARGET_COLUMN,
        output_dir=tmp_path,
        enable_pca=False,
        persist_artifacts=True,
        enable_exploration=False,
        min_class_percentage=15.0,
    )

    class_purge_metadata = result.metadata["class_purge"]
    assert class_purge_metadata["enabled"] is True
    assert class_purge_metadata["original_class_count"] == 13
    assert class_purge_metadata["retained_class_count"] == 3

    # O dataset final preparado só deve conter as classes retidas.
    retained_targets = set(result.prepared_dataset[TARGET_COLUMN].unique())
    assert len(retained_targets) == 3

    # Artefatos "antes"/"depois" da distribuição de classes devem existir em disco.
    artifact_paths = class_purge_metadata["artifact_paths"]
    for stage in ("before_purge", "after_purge"):
        for kind in ("csv", "json", "png", "html"):
            assert Path(artifact_paths[stage][kind]).exists()

    before_csv = pd.read_csv(artifact_paths["before_purge"]["csv"])
    after_csv = pd.read_csv(artifact_paths["after_purge"]["csv"])
    assert before_csv.shape[0] == 13
    assert after_csv.shape[0] == 3


def test_prepare_training_dataset_can_disable_class_purge(
    synthetic_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    result = prepare_training_dataset(
        data_frame=synthetic_dataset,
        target_column=TARGET_COLUMN,
        output_dir=tmp_path,
        enable_pca=False,
        persist_artifacts=False,
        enable_exploration=False,
        enable_class_purge=False,
    )

    assert result.metadata["class_purge"]["enabled"] is False
    retained_targets = set(result.prepared_dataset[TARGET_COLUMN].unique())
    assert len(retained_targets) == 13
