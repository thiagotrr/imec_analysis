import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler

try:
    from .feature_engineering import (
        DEFAULT_TARGET_COLUMN,
        FeatureRecommendations,
        get_analysis_bundle,
        get_feature_recommendations,
        load_dataset,
        resolve_dataset_path,
    )
except ImportError:
    from feature_engineering import (
        DEFAULT_TARGET_COLUMN,
        FeatureRecommendations,
        get_analysis_bundle,
        get_feature_recommendations,
        load_dataset,
        resolve_dataset_path,
    )


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
DEFAULT_PIPELINE_FILENAME = "preprocessing_pipeline.pkl"
DEFAULT_TARGET_ENCODER_FILENAME = "target_encoder.pkl"
DEFAULT_METADATA_FILENAME = "preparation_metadata.json"
DEFAULT_PREPARED_DATASET_FILENAME = "prepared_training_dataset.csv"
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_SPARSE_COMPONENTS = 128


@dataclass(frozen=True)
class PreparationArtifacts:
    model_dir: Path
    pipeline_path: Path
    metadata_path: Path
    prepared_dataset_path: Path
    target_encoder_path: Path | None = None


@dataclass(frozen=True)
class PreparedDatasetResult:
    source_path: Path
    target_column: str
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    feature_shape_before_transform: tuple[int, int]
    transformed_shape: tuple[int, int]
    prepared_dataset: pd.DataFrame
    target: pd.Series
    recommendations: FeatureRecommendations
    feature_groups: dict[str, list[str]]
    dropped_columns: dict[str, list[str]]
    pipeline: Pipeline
    metadata: dict[str, object]
    artifacts: PreparationArtifacts | None = None


def clone_dataset(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.copy(deep=True)


def _normalize_string_value(value: object) -> object:
    if not isinstance(value, str):
        return value

    cleaned_value = value.strip()
    return cleaned_value if cleaned_value else pd.NA


def normalize_string_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    normalized_frame = clone_dataset(data_frame)

    for column in normalized_frame.select_dtypes(include=["object", "string", "category"]).columns:
        normalized_frame[column] = normalized_frame[column].map(_normalize_string_value)

    return normalized_frame


def _remove_duplicate_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.drop_duplicates().reset_index(drop=True)


def drop_constant_columns(feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    constant_columns = [
        column for column in feature_frame.columns
        if feature_frame[column].nunique(dropna=False) <= 1
    ]
    return feature_frame.drop(columns=constant_columns, errors="ignore"), constant_columns


def drop_highly_correlated_columns(
    feature_frame: pd.DataFrame,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    numeric_frame = feature_frame.select_dtypes(include=["number", "bool"])
    if numeric_frame.shape[1] < 2:
        return feature_frame, []

    correlation_matrix = numeric_frame.astype(float).corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_correlation_columns = [
        column for column in upper_triangle.columns
        if (upper_triangle[column] > threshold).any()
    ]
    return feature_frame.drop(columns=high_correlation_columns, errors="ignore"), high_correlation_columns


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _stringify_values(values: object) -> np.ndarray:
    return np.asarray(values).astype(str)


def _split_feature_groups(
    feature_frame: pd.DataFrame,
    recommendations: FeatureRecommendations,
) -> tuple[list[str], list[str], list[str]]:
    numeric_columns = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = feature_frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    scale_columns = [column for column in recommendations.cols_to_scale if column in numeric_columns]
    passthrough_numeric_columns = [
        column for column in numeric_columns
        if column not in scale_columns
    ]
    return scale_columns, categorical_columns, passthrough_numeric_columns


def _build_column_transformer(
    scale_columns: list[str],
    categorical_columns: list[str],
    passthrough_numeric_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if scale_columns:
        transformers.append(
            (
                "scaled_numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                scale_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "encoded_categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("stringify", FunctionTransformer(_stringify_values, validate=False)),
                        ("encoder", _build_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    if passthrough_numeric_columns:
        transformers.append(
            (
                "passthrough_numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                passthrough_numeric_columns,
            )
        )

    if not transformers:
        raise ValueError("Nenhuma feature elegível restou após a limpeza do dataset.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def _build_dimensionality_reducer(
    transformed_probe: object,
    pca_components: float | int,
) -> PCA | TruncatedSVD | None:
    transformed_width = transformed_probe.shape[1]
    if transformed_width <= 1:
        return None

    if sparse.issparse(transformed_probe) or (
        isinstance(pca_components, float) and 0 < pca_components < 1 and transformed_width > 1000
    ):
        if isinstance(pca_components, int):
            n_components = max(1, min(pca_components, transformed_width - 1))
        else:
            n_components = max(2, min(DEFAULT_SPARSE_COMPONENTS, transformed_width - 1))
        return TruncatedSVD(n_components=n_components, random_state=42)

    return PCA(n_components=pca_components)


def build_preprocessing_pipeline(
    feature_frame: pd.DataFrame,
    recommendations: FeatureRecommendations,
    enable_pca: bool = True,
    pca_components: float | int = 0.95,
) -> tuple[Pipeline, dict[str, list[str]]]:
    scale_columns, categorical_columns, passthrough_numeric_columns = _split_feature_groups(
        feature_frame,
        recommendations,
    )
    preprocessor = _build_column_transformer(
        scale_columns,
        categorical_columns,
        passthrough_numeric_columns,
    )

    pipeline_steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]
    if enable_pca:
        pipeline_steps.append(("pca", PCA(n_components=pca_components)))

    feature_groups = {
        "scaled_columns": scale_columns,
        "categorical_columns": categorical_columns,
        "passthrough_numeric_columns": passthrough_numeric_columns,
    }
    return Pipeline(steps=pipeline_steps), feature_groups


def fit_preprocessor(
    feature_frame: pd.DataFrame,
    recommendations: FeatureRecommendations,
    enable_pca: bool = True,
    pca_components: float | int = 0.95,
) -> tuple[Pipeline, dict[str, list[str]]]:
    probe_pipeline, feature_groups = build_preprocessing_pipeline(
        feature_frame,
        recommendations,
        enable_pca=False,
        pca_components=pca_components,
    )
    transformed_probe = probe_pipeline.fit_transform(feature_frame)
    if hasattr(transformed_probe, "toarray"):
        transformed_probe = transformed_probe.toarray()

    transformed_width = transformed_probe.shape[1]
    if not enable_pca or transformed_width <= 1:
        return probe_pipeline, feature_groups

    reducer = _build_dimensionality_reducer(transformed_probe, pca_components)
    preprocessor = probe_pipeline.named_steps["preprocessor"]
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("dimensionality_reduction", reducer),
        ]
    )
    final_pipeline.fit(feature_frame)
    return final_pipeline, feature_groups


def _get_transformed_feature_names(pipeline: Pipeline, transformed_width: int) -> list[str]:
    if "dimensionality_reduction" in pipeline.named_steps:
        reducer = pipeline.named_steps["dimensionality_reduction"]
        prefix = "svd" if isinstance(reducer, TruncatedSVD) else "pca"
        return [f"{prefix}_{index:03d}" for index in range(1, transformed_width + 1)]

    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out().tolist()

    return [f"feature_{index:03d}" for index in range(1, transformed_width + 1)]


def transform_dataset(feature_frame: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    transformed = pipeline.transform(feature_frame)
    if sparse.issparse(transformed):
        feature_names = _get_transformed_feature_names(pipeline, transformed.shape[1])
        return pd.DataFrame.sparse.from_spmatrix(transformed, index=feature_frame.index, columns=feature_names)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)
    feature_names = _get_transformed_feature_names(pipeline, transformed.shape[1])
    return pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)


def _prepare_target_series(target_series: pd.Series) -> tuple[pd.Series, LabelEncoder | None]:
    target_copy = target_series.map(_normalize_string_value)

    if isinstance(target_copy.dtype, pd.CategoricalDtype):
        encoder = LabelEncoder()
        encoded_target = encoder.fit_transform(target_copy.astype(str))
        return pd.Series(encoded_target, index=target_series.index, name=target_series.name), encoder

    if pd.api.types.is_object_dtype(target_copy) or pd.api.types.is_string_dtype(target_copy):
        encoder = LabelEncoder()
        encoded_target = encoder.fit_transform(target_copy.astype(str))
        return pd.Series(encoded_target, index=target_series.index, name=target_series.name), encoder

    if pd.api.types.is_bool_dtype(target_copy):
        return target_copy.astype(int), None

    return target_copy, None


def get_model_dir(output_dir: str | Path | None = None) -> Path:
    return Path(output_dir) if output_dir is not None else DEFAULT_MODEL_DIR


def save_preprocessing_artifacts(
    prepared_dataset: pd.DataFrame,
    pipeline: Pipeline,
    metadata: dict[str, object],
    target_encoder: LabelEncoder | None = None,
    output_dir: str | Path | None = None,
) -> PreparationArtifacts:
    model_dir = get_model_dir(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / DEFAULT_PIPELINE_FILENAME
    metadata_path = model_dir / DEFAULT_METADATA_FILENAME
    prepared_dataset_path = model_dir / DEFAULT_PREPARED_DATASET_FILENAME
    target_encoder_path = model_dir / DEFAULT_TARGET_ENCODER_FILENAME if target_encoder is not None else None

    metadata_to_save = dict(metadata)
    metadata_to_save["artifacts"] = {
        "pipeline_path": str(pipeline_path),
        "metadata_path": str(metadata_path),
        "prepared_dataset_path": str(prepared_dataset_path),
        "target_encoder_path": str(target_encoder_path) if target_encoder_path is not None else None,
    }

    with pipeline_path.open("wb") as file_obj:
        pickle.dump(pipeline, file_obj)

    if target_encoder_path is not None:
        with target_encoder_path.open("wb") as file_obj:
            pickle.dump(target_encoder, file_obj)

    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata_to_save, file_obj, indent=2, ensure_ascii=False)

    prepared_dataset.to_csv(prepared_dataset_path, index=False)

    return PreparationArtifacts(
        model_dir=model_dir,
        pipeline_path=pipeline_path,
        metadata_path=metadata_path,
        prepared_dataset_path=prepared_dataset_path,
        target_encoder_path=target_encoder_path,
    )


def prepare_training_dataset(
    data_frame: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    output_dir: str | Path | None = None,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    enable_pca: bool = True,
    pca_components: float | int = 0.95,
    persist_artifacts: bool = True,
) -> PreparedDatasetResult:
    source_path = resolve_dataset_path(dataset_path)
    raw_frame = load_dataset(source_path) if data_frame is None else clone_dataset(data_frame)
    original_shape = raw_frame.shape

    if target_column not in raw_frame.columns:
        raise ValueError(f"A coluna-alvo '{target_column}' nao existe no dataset.")

    normalized_frame = normalize_string_columns(raw_frame)
    deduplicated_frame = _remove_duplicate_rows(normalized_frame)
    target_null_rows_removed = int(deduplicated_frame[target_column].isna().sum())
    cleaned_frame = deduplicated_frame.dropna(subset=[target_column]).reset_index(drop=True)

    _, recommendations = get_analysis_bundle(
        cleaned_frame,
        dataset_path=source_path,
        target_column=target_column,
    )

    heuristic_drop_columns = [
        column for column in recommendations.cols_to_drop
        if column in cleaned_frame.columns and column != target_column
    ]

    feature_frame = cleaned_frame.drop(
        columns=heuristic_drop_columns + [target_column],
        errors="ignore",
    ).copy()
    feature_frame, constant_columns = drop_constant_columns(feature_frame)
    feature_frame, high_correlation_columns = drop_highly_correlated_columns(
        feature_frame,
        threshold=correlation_threshold,
    )

    pipeline_recommendations = get_feature_recommendations(
        pd.concat([feature_frame, cleaned_frame[[target_column]]], axis=1),
        target_column=target_column,
    )
    pipeline, feature_groups = fit_preprocessor(
        feature_frame,
        pipeline_recommendations,
        enable_pca=enable_pca,
        pca_components=pca_components,
    )

    transformed_features = transform_dataset(feature_frame, pipeline)
    transformed_target, target_encoder = _prepare_target_series(cleaned_frame[target_column].copy())

    prepared_dataset = transformed_features.copy()
    prepared_dataset.insert(0, target_column, transformed_target.to_numpy())

    reduction_step = pipeline.named_steps.get("dimensionality_reduction")
    explained_variance = [
        float(value) for value in getattr(reduction_step, "explained_variance_ratio_", [])
    ]
    metadata: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_path": str(source_path),
        "target_column": target_column,
        "original_shape": list(original_shape),
        "cleaned_shape": list(cleaned_frame.shape),
        "feature_shape_before_transform": list(feature_frame.shape),
        "transformed_shape": list(transformed_features.shape),
        "duplicate_rows_removed": int(original_shape[0] - deduplicated_frame.shape[0]),
        "target_null_rows_removed": target_null_rows_removed,
        "dropped_columns": {
            "heuristic": heuristic_drop_columns,
            "constant": constant_columns,
            "high_correlation": high_correlation_columns,
        },
        "feature_groups": feature_groups,
        "retained_feature_columns": feature_frame.columns.tolist(),
        "output_feature_columns": transformed_features.columns.tolist(),
        "heuristic_recommendations": {
            "cols_to_drop": recommendations.cols_to_drop,
            "cols_to_scale": recommendations.cols_to_scale,
            "cols_to_encode": recommendations.cols_to_encode,
        },
        "dimensionality_reduction": {
            "enabled": reduction_step is not None,
            "method": reduction_step.__class__.__name__ if reduction_step is not None else None,
            "parameter": pca_components if enable_pca else None,
            "n_components": int(getattr(reduction_step, "n_components_", transformed_features.shape[1])) if reduction_step is not None else None,
            "explained_variance_ratio": explained_variance,
            "explained_variance_ratio_sum": float(sum(explained_variance)) if explained_variance else None,
        },
        "target": {
            "dtype": str(cleaned_frame[target_column].dtype),
            "encoded": target_encoder is not None,
            "classes": target_encoder.classes_.tolist() if target_encoder is not None else None,
        },
    }

    artifacts = None
    if persist_artifacts:
        artifacts = save_preprocessing_artifacts(
            prepared_dataset=prepared_dataset,
            pipeline=pipeline,
            metadata=metadata,
            target_encoder=target_encoder,
            output_dir=output_dir,
        )
        metadata["artifacts"] = {
            "pipeline_path": str(artifacts.pipeline_path),
            "metadata_path": str(artifacts.metadata_path),
            "prepared_dataset_path": str(artifacts.prepared_dataset_path),
            "target_encoder_path": str(artifacts.target_encoder_path) if artifacts.target_encoder_path else None,
        }

    return PreparedDatasetResult(
        source_path=source_path,
        target_column=target_column,
        original_shape=original_shape,
        cleaned_shape=cleaned_frame.shape,
        feature_shape_before_transform=feature_frame.shape,
        transformed_shape=transformed_features.shape,
        prepared_dataset=prepared_dataset,
        target=transformed_target,
        recommendations=recommendations,
        feature_groups=feature_groups,
        dropped_columns={
            "heuristic": heuristic_drop_columns,
            "constant": constant_columns,
            "high_correlation": high_correlation_columns,
        },
        pipeline=pipeline,
        metadata=metadata,
        artifacts=artifacts,
    )


def print_preparation_summary(result: PreparedDatasetResult) -> None:
    print(f"\nDataset origem: {result.source_path.name}")
    print(f"Target: {result.target_column}")
    print(f"Shape original: {result.original_shape}")
    print(f"Shape após limpeza: {result.cleaned_shape}")
    print(f"Shape final preparado: {result.prepared_dataset.shape}")
    print(f"Colunas removidas por heurística: {len(result.dropped_columns['heuristic'])}")
    print(f"Colunas removidas por constância: {len(result.dropped_columns['constant'])}")
    print(f"Colunas removidas por correlação: {len(result.dropped_columns['high_correlation'])}")

    reduction_metadata = result.metadata["dimensionality_reduction"]
    if reduction_metadata["enabled"]:
        print(
            "Redução dimensional habilitada: "
            f"{reduction_metadata['method']} com {reduction_metadata['n_components']} componentes, "
            f"variância explicada acumulada={reduction_metadata['explained_variance_ratio_sum']:.4f}"
        )

    if result.artifacts is not None:
        print(f"Artefatos: {result.artifacts.model_dir}")
        print(f"Pipeline: {result.artifacts.pipeline_path.name}")
        print(f"Metadata: {result.artifacts.metadata_path.name}")
        print(f"Dataset preparado: {result.artifacts.prepared_dataset_path.name}")
        if result.artifacts.target_encoder_path is not None:
            print(f"Encoder target: {result.artifacts.target_encoder_path.name}")


if __name__ == "__main__":
    print_preparation_summary(prepare_training_dataset())
