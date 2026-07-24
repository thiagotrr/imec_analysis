from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_FILENAME = "resultado_laudo_afericao.xlsx"
DEFAULT_DATASET_PATH = Path(__file__).parent / DEFAULT_DATASET_FILENAME
DEFAULT_TARGET_COLUMN = "CODRSTAFER"

MANUALLY_REMOVED_FEATURES = [
    "VLRENS_CGA_NMN",
    "VLRENS_CGA_CPC",
    "RESPAFER",
    "CODPRSERV",
    "VLRDVI_ELM_A",
    "VLRDVI_ELM_B",
    "VLRDVI_ELM_C",
    "FTRCRC_CGA_NMN_ELM",
    "TPRIFR",
    "TPRSUP",
    "FTRCRC_CGA_NMN_ELM_1",
    "TPRIFR_1",
    "TPRSUP_1",
    "CODPRJ",
    "ENSAIO_LAUDO_CORR",
    "INDRST_ENS_COR",
    "INDRST_ENS_TNS",
    "ENSAIO_LAUDO_MESA",
    "ENSAIO_LAUDO_TEMPERATURA",
    "ENSAIO_LAUDO_TEMPERATURA2",
    "IND_LAUDO_EXTERNO",
    "VLRENS_CGA_CPC_1",
    "VLRDVI_ELM_B_1",
    "VLRDVI_ELM_C_1",
    "INDRST_ENS_COR_1",
    "INDRST_ENS_TNS_1",
    "VLRLTR_MAN_KWH",
    "VLRLTR_MAN_KVARH",
    "VLR_LTR_MAN_KVARH",
    "SEQ_NUMLAUDO",
]

HIGH_NULL_THRESHOLD = 50.0
FREE_TEXT_AVG_LEN_THRESHOLD = 30
FREE_TEXT_CARDINALITY_RATIO = 0.80

MIN_CLASS_PERCENTAGE_THRESHOLD = 15.0
"""Percentual mínimo (sobre o total de linhas) que uma classe do target
precisa representar para ser mantida no dataset de treino/teste.

Classes abaixo deste limiar são consideradas de baixa representatividade e
removidas antes do treino, tanto na etapa de preparação
(``data_preparation.prepare_training_dataset``) quanto na etapa de split de
classificação (``classification.workflow._split_dataset``), via
``filter_classes_by_percentage``. Essa é a estratégia primária de expurgo,
que substitui o antigo critério baseado apenas em contagem absoluta
(``min_class_count``); a contagem absoluta é mantida como rede de segurança
complementar para eventuais classes residuais muito raras.

Como o problema tem dezenas de classes (a maioria naturalmente com fração
individual pequena), um limiar de 15% tende a reter apenas as classes
dominantes — esse é o comportamento intencional solicitado para focar o
modelo nas classes de maior representatividade, em detrimento de classes
raras que os modelos historicamente classificam muito mal.
"""


@dataclass(frozen=True)
class DatasetProfile:
    dataset_path: Path
    n_rows: int
    n_cols: int
    types_by_col: pd.Series
    type_groups: dict[str, list[str]]
    null_counts: pd.Series
    null_pct: pd.Series
    high_null_cols: list[str]
    free_text_cols: list[str]
    id_like_cols: list[str]
    datetime_cols: list[str]


@dataclass(frozen=True)
class FeatureRecommendations:
    target_column: str | None
    cols_to_drop: list[str]
    cols_to_scale: list[str]
    cols_to_encode: list[str]
    numeric_cols: list[str]
    object_cols: list[str]


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    return Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_excel(resolve_dataset_path(dataset_path))


def _group_types(types_by_col: pd.Series) -> dict[str, list[str]]:
    type_groups: dict[str, list[str]] = {}
    for col, dtype in types_by_col.items():
        type_groups.setdefault(str(dtype), []).append(col)
    return type_groups


def detect_free_text_columns(data_frame: pd.DataFrame) -> list[str]:
    n_rows = max(len(data_frame), 1)
    free_text_cols: list[str] = []

    for col in data_frame.select_dtypes(include=["object", "string", "category"]).columns:
        cleaned_values = data_frame[col].dropna().astype(str).str.strip()
        if cleaned_values.empty:
            continue

        avg_len = cleaned_values.str.len().mean()
        cardinality_ratio = cleaned_values.nunique() / n_rows
        if avg_len > FREE_TEXT_AVG_LEN_THRESHOLD or cardinality_ratio > FREE_TEXT_CARDINALITY_RATIO:
            free_text_cols.append(col)

    return free_text_cols


def get_manually_removed_features(
    data_frame: pd.DataFrame,
    target_column: str | None = DEFAULT_TARGET_COLUMN,
) -> list[str]:
    return [
        column for column in MANUALLY_REMOVED_FEATURES
        if column in data_frame.columns and column != target_column
    ]


def build_dataset_profile(
    data_frame: pd.DataFrame,
    dataset_path: str | Path | None = None,
) -> DatasetProfile:
    resolved_dataset_path = resolve_dataset_path(dataset_path)
    n_rows, n_cols = data_frame.shape
    types_by_col = data_frame.dtypes
    type_groups = _group_types(types_by_col)
    null_counts = data_frame.isnull().sum()
    null_pct = (null_counts / max(n_rows, 1) * 100).round(2)
    high_null_cols = null_pct[null_pct > HIGH_NULL_THRESHOLD].index.tolist()
    free_text_cols = detect_free_text_columns(data_frame)
    id_like_cols = [
        col for col in data_frame.columns
        if n_rows > 0 and data_frame[col].nunique(dropna=False) == n_rows
    ]
    datetime_cols = [
        col for col, dtype in types_by_col.items()
        if "datetime" in str(dtype)
    ]

    return DatasetProfile(
        dataset_path=resolved_dataset_path,
        n_rows=n_rows,
        n_cols=n_cols,
        types_by_col=types_by_col,
        type_groups=type_groups,
        null_counts=null_counts,
        null_pct=null_pct,
        high_null_cols=high_null_cols,
        free_text_cols=free_text_cols,
        id_like_cols=id_like_cols,
        datetime_cols=datetime_cols,
    )


def get_feature_recommendations(
    data_frame: pd.DataFrame,
    target_column: str | None = DEFAULT_TARGET_COLUMN,
) -> FeatureRecommendations:
    profile = build_dataset_profile(data_frame)
    numeric_cols = data_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    object_cols = data_frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    suggested_to_drop = list(
        dict.fromkeys(
            profile.high_null_cols
            + profile.id_like_cols
            + profile.datetime_cols
            + profile.free_text_cols
            + get_manually_removed_features(data_frame, target_column)
        )
    )

    if target_column in suggested_to_drop:
        suggested_to_drop.remove(target_column)

    suggested_to_scale = [
        col for col in numeric_cols
        if col not in suggested_to_drop
        and col != target_column
        and not (
            data_frame[col].dropna().isin([0, 1]).all()
            and data_frame[col].nunique(dropna=True) <= 2
        )
    ]

    suggested_to_encode = [
        col for col in object_cols
        if col not in suggested_to_drop
        and col not in profile.free_text_cols
        and col != target_column
    ]

    return FeatureRecommendations(
        target_column=target_column,
        cols_to_drop=suggested_to_drop,
        cols_to_scale=suggested_to_scale,
        cols_to_encode=suggested_to_encode,
        numeric_cols=numeric_cols,
        object_cols=object_cols,
    )


def compute_class_distribution(target_series: pd.Series) -> pd.DataFrame:
    """Calcula a contagem e o percentual de cada classe presente em ``target_series``.

    Retorna um ``DataFrame`` com colunas ``class``, ``count`` e ``percentage``
    (percentual sobre o total de linhas de ``target_series``), ordenado da
    classe mais para a menos frequente.
    """
    total_rows = len(target_series)
    counts = target_series.value_counts(dropna=False).sort_values(ascending=False)
    distribution = counts.rename("count").reset_index()
    distribution.columns = ["class", "count"]
    distribution["class"] = distribution["class"].astype(str)
    distribution["percentage"] = (distribution["count"] / max(total_rows, 1) * 100).round(4)
    return distribution[["class", "count", "percentage"]]


def identify_low_representation_classes(
    target_series: pd.Series,
    min_percentage: float = MIN_CLASS_PERCENTAGE_THRESHOLD,
) -> list[str]:
    """Retorna os rótulos (como string) das classes cujo percentual sobre o total é menor que ``min_percentage``."""
    distribution = compute_class_distribution(target_series)
    return distribution.loc[distribution["percentage"] < min_percentage, "class"].tolist()


def filter_classes_by_percentage(
    data_frame: pd.DataFrame,
    target_column: str,
    min_percentage: float = MIN_CLASS_PERCENTAGE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove linhas cuja classe do target represente menos que ``min_percentage``% do total de linhas.

    Retorna uma tupla ``(dataframe_filtrado, metadata)`` em que ``metadata``
    documenta as distribuições antes/depois do expurgo e as classes
    removidas, para fins de auditoria/relatório.
    """
    if target_column not in data_frame.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no dataset.")

    before_distribution = compute_class_distribution(data_frame[target_column])
    low_representation_classes = before_distribution.loc[
        before_distribution["percentage"] < min_percentage, "class"
    ].tolist()

    target_as_str = data_frame[target_column].astype(str)
    filtered_frame = data_frame[~target_as_str.isin(low_representation_classes)].copy()
    after_distribution = compute_class_distribution(filtered_frame[target_column])

    original_rows = len(data_frame)
    retained_rows = len(filtered_frame)
    metadata = {
        "min_class_percentage": min_percentage,
        "original_rows": int(original_rows),
        "retained_rows": int(retained_rows),
        "removed_rows": int(original_rows - retained_rows),
        "retained_rows_percentage": round(retained_rows / max(original_rows, 1) * 100, 4),
        "original_class_count": int(before_distribution.shape[0]),
        "retained_class_count": int(after_distribution.shape[0]),
        "removed_class_count": int(before_distribution.shape[0] - after_distribution.shape[0]),
        "removed_classes": before_distribution[
            before_distribution["class"].isin(low_representation_classes)
        ].to_dict(orient="records"),
        "before_distribution": before_distribution.to_dict(orient="records"),
        "after_distribution": after_distribution.to_dict(orient="records"),
    }
    return filtered_frame, metadata


def get_analysis_bundle(
    data_frame: pd.DataFrame,
    dataset_path: str | Path | None = None,
    target_column: str | None = DEFAULT_TARGET_COLUMN,
) -> tuple[DatasetProfile, FeatureRecommendations]:
    profile = build_dataset_profile(data_frame, dataset_path)
    recommendations = get_feature_recommendations(data_frame, target_column)
    return profile, recommendations


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_report(
    data_frame: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
    target_column: str | None = DEFAULT_TARGET_COLUMN,
) -> None:
    resolved_dataset_path = resolve_dataset_path(dataset_path)
    working_frame = data_frame if data_frame is not None else load_dataset(resolved_dataset_path)
    profile, recommendations = get_analysis_bundle(
        working_frame,
        dataset_path=resolved_dataset_path,
        target_column=target_column,
    )

    print(f"\nDataset: {profile.dataset_path.name}")
    print(f"Dimensões: {profile.n_rows} linhas × {profile.n_cols} colunas")

    print(working_frame.head(10))

    _section("TIPOS DE DADOS POR COLUNA")
    for dtype, cols in profile.type_groups.items():
        print(f"\n  [{dtype}] ({len(cols)} colunas)")
        for col in cols:
            print(f"    - {col}")

    _section("ANÁLISE DE NULOS")
    cols_with_nulls = profile.null_counts[profile.null_counts > 0]
    if cols_with_nulls.empty:
        print("\n  Nenhuma coluna com valores nulos.")
    else:
        print(f"\n  {'Coluna':<40} {'Nulos':>8}  {'%':>7}")
        print(f"  {'-' * 57}")
        for col in cols_with_nulls.index:
            flag = "  <- >50%" if col in profile.high_null_cols else ""
            print(
                f"  {col:<40} {int(profile.null_counts[col]):>8}"
                f"  {profile.null_pct[col]:>6.1f}%{flag}"
            )

    _section("COLUNAS DE TEXTO LIVRE (detectadas)")
    if profile.free_text_cols:
        for col in profile.free_text_cols:
            cleaned_values = working_frame[col].dropna().astype(str).str.strip()
            avg_len = cleaned_values.str.len().mean()
            card_ratio = cleaned_values.nunique() / max(profile.n_rows, 1)
            print(f"  - {col:<38}  avg_len={avg_len:.1f}  card_ratio={card_ratio:.2f}")
    else:
        print("\n  Nenhuma coluna de texto livre detectada.")

    _section("ARRAYS SUGERIDOS (heurística)")
    print(f"\n  COLS_TO_DROP  ({len(recommendations.cols_to_drop)}): {recommendations.cols_to_drop}")
    print(f"\n  COLS_TO_SCALE ({len(recommendations.cols_to_scale)}): {recommendations.cols_to_scale}")
    print(f"\n  COLS_TO_ENCODE ({len(recommendations.cols_to_encode)}): {recommendations.cols_to_encode}")
    print()


if __name__ == "__main__":
    print_report()
