import json
import pickle
import webbrowser
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
    from .data_exploration import (
        compute_numeric_correlation_matrix,
        generate_exploration_artifacts,
        save_class_distribution_artifacts,
    )
    from .feature_engineering import (
        DEFAULT_TARGET_COLUMN,
        MIN_CLASS_PERCENTAGE_THRESHOLD,
        DatasetProfile,
        FeatureRecommendations,
        build_class_weight_registry,
        compute_class_distribution,
        filter_classes_by_percentage,
        get_analysis_bundle,
        get_feature_recommendations,
        load_dataset,
        qualify_class_distribution,
        resolve_dataset_path,
    )
except ImportError:
    from data_exploration import (
        compute_numeric_correlation_matrix,
        generate_exploration_artifacts,
        save_class_distribution_artifacts,
    )
    from feature_engineering import (
        DEFAULT_TARGET_COLUMN,
        MIN_CLASS_PERCENTAGE_THRESHOLD,
        DatasetProfile,
        FeatureRecommendations,
        build_class_weight_registry,
        compute_class_distribution,
        filter_classes_by_percentage,
        get_analysis_bundle,
        get_feature_recommendations,
        load_dataset,
        qualify_class_distribution,
        resolve_dataset_path,
    )


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
DEFAULT_PIPELINE_FILENAME = "preprocessing_pipeline.pkl"
DEFAULT_TARGET_ENCODER_FILENAME = "target_encoder.pkl"
DEFAULT_METADATA_FILENAME = "preparation_metadata.json"
DEFAULT_PREPARED_DATASET_FILENAME = "prepared_training_dataset.csv"
DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME = "class_weight_registry.json"
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_SPARSE_COMPONENTS = 128


@dataclass(frozen=True)
class PreparationArtifacts:
    model_dir: Path
    pipeline_path: Path
    metadata_path: Path
    prepared_dataset_path: Path
    target_encoder_path: Path | None = None
    class_weight_registry_path: Path | None = None


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


@dataclass(frozen=True)
class PreparationWorkflowResult:
    preparation_result: PreparedDatasetResult
    dashboard_path: Path | None = None


def _as_profile_metadata(profile: DatasetProfile) -> dict[str, object]:
    return {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "high_null_cols": profile.high_null_cols,
        "free_text_cols": profile.free_text_cols,
        "id_like_cols": profile.id_like_cols,
        "datetime_cols": profile.datetime_cols,
        "type_groups": profile.type_groups,
    }


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
    correlation_matrix = compute_numeric_correlation_matrix(feature_frame)
    if correlation_matrix.shape[1] < 2:
        return feature_frame, []

    upper_triangle = correlation_matrix.abs().where(
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
                        (
                            "stringify",
                            FunctionTransformer(_stringify_values, validate=False, feature_names_out="one-to-one"),
                        ),
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
    # Mantemos `transformed_probe` no formato original (esparso ou denso)
    # retornado pelo `ColumnTransformer`: `_build_dimensionality_reducer`
    # decide entre `TruncatedSVD` (para entradas esparsas, ex.: quando há
    # colunas categóricas com one-hot encoding) e `PCA` (para entradas densas)
    # justamente com base em `sparse.issparse(transformed_probe)`. Densificar
    # aqui antes dessa checagem faria a decisão sempre cair em `PCA`, que não
    # suporta `n_components` fracionário (variância explicada) em entradas
    # esparsas — foi exatamente o bug corrigido nesta revisão.
    transformed_probe = probe_pipeline.fit_transform(feature_frame)

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


def _sparse_matrix_to_frame(transformed: object, index: pd.Index, feature_names: list[str]) -> pd.DataFrame:
    # `pd.DataFrame.sparse.from_spmatrix` usa `fill_value=NaN` por padrão nesta
    # versão do pandas, o que representa incorretamente o "zero implícito" da
    # matriz esparsa (ex.: categoria não-selecionada do one-hot encoding) como
    # valor ausente — e `DataFrame.astype(SparseDtype(..., fill_value=0.0))`
    # não corrige isso, pois a igualdade de `SparseDtype` ignora `fill_value`
    # e o astype vira um no-op. Construímos cada coluna explicitamente com
    # `fill_value=0.0` para preservar a esparsidade sem introduzir NaN
    # espúrios (o que quebraria consumidores como SMOTE/ADASYN e o round-trip
    # via CSV).
    csc_matrix = transformed.tocsc()
    sparse_columns = {
        name: pd.arrays.SparseArray(csc_matrix.getcol(position).toarray().ravel(), fill_value=0.0)
        for position, name in enumerate(feature_names)
    }
    return pd.DataFrame(sparse_columns, index=index)


def transform_dataset(feature_frame: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    transformed = pipeline.transform(feature_frame)
    if sparse.issparse(transformed):
        feature_names = _get_transformed_feature_names(pipeline, transformed.shape[1])
        return _sparse_matrix_to_frame(transformed, feature_frame.index, feature_names)

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


def save_class_weight_registry(
    registry: dict[str, object],
    output_dir: str | Path | None = None,
) -> Path:
    """Persiste o "registro de pesos e qualificação de classes" (ver ``feature_engineering.build_class_weight_registry``).

    Salvo em ``model/class_weight_registry.json`` (não em ``model/exploration``,
    que é tratado como saída descartável no ``.gitignore``): este é um
    artefato estável da aplicação, versionado no repositório, pensado para
    ser consumido por uma futura camada de inferência/API.
    """
    model_dir = get_model_dir(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    registry_path = model_dir / DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **registry,
    }
    with registry_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False, default=str)
    return registry_path


def save_preprocessing_artifacts(
    prepared_dataset: pd.DataFrame,
    pipeline: Pipeline,
    metadata: dict[str, object],
    target_encoder: LabelEncoder | None = None,
    output_dir: str | Path | None = None,
    class_weight_registry: dict[str, object] | None = None,
) -> PreparationArtifacts:
    model_dir = get_model_dir(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / DEFAULT_PIPELINE_FILENAME
    metadata_path = model_dir / DEFAULT_METADATA_FILENAME
    prepared_dataset_path = model_dir / DEFAULT_PREPARED_DATASET_FILENAME
    target_encoder_path = model_dir / DEFAULT_TARGET_ENCODER_FILENAME if target_encoder is not None else None
    class_weight_registry_path = (
        save_class_weight_registry(class_weight_registry, output_dir) if class_weight_registry is not None else None
    )

    metadata_to_save = dict(metadata)
    metadata_to_save["artifacts"] = {
        "pipeline_path": str(pipeline_path),
        "metadata_path": str(metadata_path),
        "prepared_dataset_path": str(prepared_dataset_path),
        "target_encoder_path": str(target_encoder_path) if target_encoder_path is not None else None,
        "class_weight_registry_path": str(class_weight_registry_path) if class_weight_registry_path is not None else None,
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
        class_weight_registry_path=class_weight_registry_path,
    )


def run_preparation_workflow(
    data_frame: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    output_dir: str | Path | None = None,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    enable_pca: bool = True,
    pca_components: float | int = 0.95,
    persist_artifacts: bool = True,
    enable_exploration: bool = True,
    open_browser: bool = True,
    enable_class_purge: bool = True,
    min_class_percentage: float = MIN_CLASS_PERCENTAGE_THRESHOLD,
) -> PreparationWorkflowResult:
    result = prepare_training_dataset(
        data_frame=data_frame,
        dataset_path=dataset_path,
        target_column=target_column,
        output_dir=output_dir,
        correlation_threshold=correlation_threshold,
        enable_pca=enable_pca,
        pca_components=pca_components,
        persist_artifacts=persist_artifacts,
        enable_exploration=enable_exploration,
        enable_class_purge=enable_class_purge,
        min_class_percentage=min_class_percentage,
    )

    dashboard_path_raw = result.metadata.get("exploration", {}).get("dashboard_path")
    dashboard_path = Path(dashboard_path_raw) if isinstance(dashboard_path_raw, str) and dashboard_path_raw else None

    if open_browser and dashboard_path is not None and dashboard_path.exists():
        webbrowser.open(dashboard_path.resolve().as_uri())

    return PreparationWorkflowResult(
        preparation_result=result,
        dashboard_path=dashboard_path,
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
    enable_exploration: bool = True,
    enable_class_purge: bool = True,
    min_class_percentage: float = MIN_CLASS_PERCENTAGE_THRESHOLD,
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

    # Distribuição de classes do target ANTES de qualquer expurgo, para documentar o "antes".
    # Já qualificada por camada (A/B/C/D) para que os artefatos "antes/depois" (CSV/JSON/PNG)
    # tragam a camada de cada classe, de forma escalável e auditável (ver
    # feature_engineering.CLASS_TIER_THRESHOLDS).
    class_distribution_before = qualify_class_distribution(compute_class_distribution(cleaned_frame[target_column]))
    if enable_class_purge:
        training_frame, class_purge_metadata = filter_classes_by_percentage(
            cleaned_frame, target_column, min_percentage=min_class_percentage
        )
    else:
        training_frame = cleaned_frame
        class_purge_metadata = None
    class_distribution_after = qualify_class_distribution(compute_class_distribution(training_frame[target_column]))

    # Registro de pesos/qualificação das classes retidas (camadas A/B/C) — insumo
    # estável da aplicação, independente de `enable_class_purge`/`persist_artifacts`
    # (sempre computado a partir da distribuição completa, antes do expurgo).
    class_weight_registry = build_class_weight_registry(
        cleaned_frame, target_column, min_percentage=min_class_percentage
    )

    class_distribution_artifact_paths: dict[str, dict[str, str]] = {}
    if persist_artifacts:
        class_distribution_artifact_paths["before_purge"] = save_class_distribution_artifacts(
            class_distribution_before, output_dir, target_column, stage="before_purge"
        )
        class_distribution_artifact_paths["after_purge"] = save_class_distribution_artifacts(
            class_distribution_after, output_dir, target_column, stage="after_purge"
        )

    profile, recommendations = get_analysis_bundle(
        cleaned_frame,
        dataset_path=source_path,
        target_column=target_column,
    )

    exploration_artifacts = None
    if enable_exploration:
        exploration_artifacts = generate_exploration_artifacts(
            cleaned_frame,
            profile=profile,
            target_column=target_column,
            output_dir=output_dir,
        )

    heuristic_drop_columns = [
        column for column in recommendations.cols_to_drop
        if column in training_frame.columns and column != target_column
    ]

    feature_frame = training_frame.drop(
        columns=heuristic_drop_columns + [target_column],
        errors="ignore",
    ).copy()
    feature_frame, constant_columns = drop_constant_columns(feature_frame)
    feature_frame, high_correlation_columns = drop_highly_correlated_columns(
        feature_frame,
        threshold=correlation_threshold,
    )

    pipeline_recommendations = get_feature_recommendations(
        pd.concat([feature_frame, training_frame[[target_column]]], axis=1),
        target_column=target_column,
    )
    pipeline, feature_groups = fit_preprocessor(
        feature_frame,
        pipeline_recommendations,
        enable_pca=enable_pca,
        pca_components=pca_components,
    )

    transformed_features = transform_dataset(feature_frame, pipeline)
    transformed_target, target_encoder = _prepare_target_series(training_frame[target_column].copy())

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
        "training_shape": list(training_frame.shape),
        "class_purge": {
            "enabled": enable_class_purge,
            "min_class_percentage": min_class_percentage,
            **(class_purge_metadata or {}),
            "artifact_paths": class_distribution_artifact_paths or None,
        },
        "class_weight_registry": class_weight_registry,
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
        "dataset_profile": _as_profile_metadata(profile),
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
        "exploration": {
            "enabled": enable_exploration,
            "output_dir": str(exploration_artifacts.output_dir) if exploration_artifacts is not None else None,
            "metadata_path": str(exploration_artifacts.metadata_path) if exploration_artifacts is not None else None,
            "dashboard_path": str(exploration_artifacts.dashboard_path) if exploration_artifacts is not None else None,
            "numeric_columns": exploration_artifacts.numeric_columns if exploration_artifacts is not None else [],
            "skipped_columns": exploration_artifacts.skipped_columns if exploration_artifacts is not None else [],
            "files_by_kind": exploration_artifacts.files_by_kind if exploration_artifacts is not None else {},
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
            class_weight_registry=class_weight_registry,
        )
        metadata["artifacts"] = {
            "pipeline_path": str(artifacts.pipeline_path),
            "metadata_path": str(artifacts.metadata_path),
            "prepared_dataset_path": str(artifacts.prepared_dataset_path),
            "target_encoder_path": str(artifacts.target_encoder_path) if artifacts.target_encoder_path else None,
            "class_weight_registry_path": (
                str(artifacts.class_weight_registry_path) if artifacts.class_weight_registry_path else None
            ),
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

    class_purge_metadata = result.metadata.get("class_purge", {})
    if class_purge_metadata.get("enabled"):
        print(
            "Expurgo de classes de baixa representatividade "
            f"(limiar={class_purge_metadata.get('min_class_percentage')}%): "
            f"{class_purge_metadata.get('original_class_count')} -> "
            f"{class_purge_metadata.get('retained_class_count')} classes, "
            f"{class_purge_metadata.get('retained_rows_percentage')}% das linhas retidas."
        )

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
    workflow_result = run_preparation_workflow()
    print_preparation_summary(workflow_result.preparation_result)
    if workflow_result.dashboard_path is not None:
        print(f"Dashboard HTML: {workflow_result.dashboard_path}")
