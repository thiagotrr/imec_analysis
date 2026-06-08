from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Carregamento do dataset
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "resultado_laudo_afericao.xlsx"
df = pd.read_excel(DATASET_PATH)

n_rows, n_cols = df.shape

# ---------------------------------------------------------------------------
# Análise de tipos de dados
# ---------------------------------------------------------------------------
types_by_col = df.dtypes

type_groups: dict[str, list[str]] = {}
for col, dtype in types_by_col.items():
    key = str(dtype)
    type_groups.setdefault(key, []).append(col)

# ---------------------------------------------------------------------------
# Análise de nulos
# ---------------------------------------------------------------------------
null_counts = df.isnull().sum()
null_pct = (null_counts / n_rows * 100).round(2)

HIGH_NULL_THRESHOLD = 50.0  # %
high_null_cols = null_pct[null_pct > HIGH_NULL_THRESHOLD].index.tolist()

# ---------------------------------------------------------------------------
# Detecção de colunas de texto livre
# ---------------------------------------------------------------------------
# Critério: colunas object com comprimento médio > 30 chars OU cardinalidade > 80% das linhas
FREE_TEXT_AVG_LEN_THRESHOLD = 30
FREE_TEXT_CARDINALITY_RATIO = 0.80

free_text_cols: list[str] = []
for col in df.select_dtypes(include=["object", "str"]).columns:
    avg_len = df[col].dropna().astype(str).str.len().mean()
    cardinality_ratio = df[col].nunique() / n_rows
    if avg_len > FREE_TEXT_AVG_LEN_THRESHOLD or cardinality_ratio > FREE_TEXT_CARDINALITY_RATIO:
        free_text_cols.append(col)

# ---------------------------------------------------------------------------
# Heurística de categorização automática
# ---------------------------------------------------------------------------
id_like_cols = [
    col for col in df.columns
    if df[col].nunique() == n_rows
]

datetime_cols = [
    col for col, dtype in types_by_col.items()
    if "datetime" in str(dtype)
]

# Sugeridas para exclusão: alta % de nulos, colunas ID, datetime, texto livre
suggested_to_drop = list(
    dict.fromkeys(high_null_cols + id_like_cols + datetime_cols + free_text_cols)
)

# Sugeridas para scaling: numéricas, não binárias (0/1), não sugeridas para drop
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
suggested_to_scale = [
    col for col in numeric_cols
    if col not in suggested_to_drop
    and not (df[col].dropna().isin([0, 1]).all() and df[col].nunique() <= 2)
]

# Sugeridas para label encoding: object, baixa cardinalidade, não texto livre, não sugeridas para drop
object_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
suggested_to_encode = [
    col for col in object_cols
    if col not in suggested_to_drop
    and col not in free_text_cols
]

# ---------------------------------------------------------------------------
# Arrays explícitos — editar manualmente após revisão do relatório
# ---------------------------------------------------------------------------
COLS_TO_DROP: list[str] = suggested_to_drop
COLS_TO_SCALE: list[str] = suggested_to_scale
COLS_TO_ENCODE: list[str] = suggested_to_encode

# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_report(data_frame: pd.DataFrame) -> None:
    print(f"\nDataset: {DATASET_PATH.name}")
    print(f"Dimensões: {n_rows} linhas × {n_cols} colunas")

    print(data_frame.head(10))

    _section("TIPOS DE DADOS POR COLUNA")
    for dtype, cols in type_groups.items():
        print(f"\n  [{dtype}] ({len(cols)} colunas)")
        for col in cols:
            print(f"    - {col}")

    _section("ANÁLISE DE NULOS")
    cols_with_nulls = null_counts[null_counts > 0]
    if cols_with_nulls.empty:
        print("\n  Nenhuma coluna com valores nulos.")
    else:
        print(f"\n  {'Coluna':<40} {'Nulos':>8}  {'%':>7}")
        print(f"  {'-' * 57}")
        for col in cols_with_nulls.index:
            flag = "  ← >50%" if col in high_null_cols else ""
            print(f"  {col:<40} {int(null_counts[col]):>8}  {null_pct[col]:>6.1f}%{flag}")

    _section("COLUNAS DE TEXTO LIVRE (detectadas)")
    if free_text_cols:
        for col in free_text_cols:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            card_ratio = df[col].nunique() / n_rows
            print(f"  - {col:<38}  avg_len={avg_len:.1f}  card_ratio={card_ratio:.2f}")
    else:
        print("\n  Nenhuma coluna de texto livre detectada.")

    _section("ARRAYS SUGERIDOS (heurística)")
    print(f"\n  COLS_TO_DROP  ({len(COLS_TO_DROP)}): {COLS_TO_DROP}")
    print(f"\n  COLS_TO_SCALE ({len(COLS_TO_SCALE)}): {COLS_TO_SCALE}")
    print(f"\n  COLS_TO_ENCODE ({len(COLS_TO_ENCODE)}): {COLS_TO_ENCODE}")
    print()


if __name__ == "__main__":
    print_report(df)
