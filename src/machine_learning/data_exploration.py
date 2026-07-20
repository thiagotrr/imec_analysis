import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots

try:
    from .feature_engineering import DatasetProfile
except ImportError:
    from feature_engineering import DatasetProfile


DEFAULT_EXPLORATION_DIRNAME = "exploration"
DEFAULT_EXPLORATION_INDEX_FILENAME = "exploration_dashboard.html"
DEFAULT_GRID_COLUMNS = 3
DEFAULT_FEATURES_PER_PAGE = 9


@dataclass(frozen=True)
class ExplorationArtifacts:
    output_dir: Path
    metadata_path: Path
    dashboard_path: Path
    files_by_kind: dict[str, list[str]]
    numeric_columns: list[str]
    skipped_columns: list[str]


def _sanitize_filename(value: str) -> str:
    safe_chars = [character.lower() if character.isalnum() else "_" for character in value.strip()]
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "target"


def get_exploration_dir(output_dir: str | Path | None = None) -> Path:
    base_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "model"
    return base_dir / DEFAULT_EXPLORATION_DIRNAME


def get_numeric_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    return feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()


def compute_numeric_correlation_matrix(data_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = data_frame.select_dtypes(include=["number", "bool"])
    if numeric_frame.empty:
        return pd.DataFrame()
    return numeric_frame.astype(float).corr()


def _partition_columns(columns: list[str], page_size: int) -> list[list[str]]:
    return [columns[index:index + page_size] for index in range(0, len(columns), page_size)]


def _build_grid(output_path: Path, total_plots: int) -> tuple[plt.Figure, list[plt.Axes]]:
    total_columns = min(DEFAULT_GRID_COLUMNS, max(total_plots, 1))
    total_rows = ceil(max(total_plots, 1) / total_columns)
    figure, axes = plt.subplots(total_rows, total_columns, figsize=(6 * total_columns, 4 * total_rows))
    flat_axes = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    return figure, flat_axes


def _save_histogram_png(data_frame: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    figure, axes = _build_grid(output_path, len(columns))
    for axis, column in zip(axes, columns):
        sns.histplot(data=data_frame, x=column, ax=axis, kde=True)
        axis.set_title(f"Histograma - {column}")
    for axis in axes[len(columns):]:
        axis.remove()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_histogram_html(data_frame: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    total_columns = min(DEFAULT_GRID_COLUMNS, max(len(columns), 1))
    total_rows = ceil(max(len(columns), 1) / total_columns)
    figure = make_subplots(rows=total_rows, cols=total_columns, subplot_titles=columns)

    for index, column in enumerate(columns):
        row = index // total_columns + 1
        col = index % total_columns + 1
        figure.add_histogram(x=data_frame[column], nbinsx=30, row=row, col=col, name=column, showlegend=False)

    figure.update_layout(title="Histogramas das features numéricas", height=360 * total_rows, width=420 * total_columns)
    figure.write_html(output_path)


def _save_scatter_png(data_frame: pd.DataFrame, target_column: str, columns: list[str], output_path: Path) -> None:
    figure, axes = _build_grid(output_path, len(columns))
    target_is_numeric = pd.api.types.is_numeric_dtype(data_frame[target_column]) or pd.api.types.is_bool_dtype(data_frame[target_column])

    for axis, column in zip(axes, columns):
        if target_is_numeric:
            sns.scatterplot(data=data_frame, x=column, y=target_column, ax=axis, s=35)
            axis.set_title(f"Dispersão - {column} x {target_column}")
        else:
            sns.boxplot(data=data_frame, x=target_column, y=column, ax=axis)
            axis.set_title(f"Distribuição - {column} por {target_column}")
            axis.tick_params(axis="x", rotation=30)

    for axis in axes[len(columns):]:
        axis.remove()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_scatter_html(data_frame: pd.DataFrame, target_column: str, columns: list[str], output_path: Path) -> None:
    total_columns = min(DEFAULT_GRID_COLUMNS, max(len(columns), 1))
    total_rows = ceil(max(len(columns), 1) / total_columns)
    figure = make_subplots(rows=total_rows, cols=total_columns, subplot_titles=columns)
    target_is_numeric = pd.api.types.is_numeric_dtype(data_frame[target_column]) or pd.api.types.is_bool_dtype(data_frame[target_column])

    for index, column in enumerate(columns):
        row = index // total_columns + 1
        col = index % total_columns + 1
        if target_is_numeric:
            trace = px.scatter(data_frame, x=column, y=target_column).data[0]
        else:
            trace = px.box(data_frame, x=target_column, y=column).data[0]
        trace.showlegend = False
        figure.add_trace(trace, row=row, col=col)

    figure.update_layout(title=f"Relação entre features numéricas e {target_column}", height=360 * total_rows, width=420 * total_columns)
    figure.write_html(output_path)


def _save_correlation_png(correlation_matrix: pd.DataFrame, output_path: Path) -> None:
    size = max(8, min(20, len(correlation_matrix.columns) * 0.8))
    figure, axis = plt.subplots(figsize=(size, size))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Matriz de correlação")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_correlation_html(correlation_matrix: pd.DataFrame, output_path: Path) -> None:
    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Matriz de correlação",
    )
    figure.write_html(output_path)


def _build_html_list_item(path: str) -> str:
        file_name = Path(path).name
        return f'<li><a href="{file_name}" target="_blank" rel="noopener noreferrer">{file_name}</a></li>'


def create_exploration_dashboard(
        output_dir: str | Path,
        target_column: str,
        files_by_kind: dict[str, list[str]],
) -> Path:
        exploration_dir = Path(output_dir)
        dashboard_path = exploration_dir / DEFAULT_EXPLORATION_INDEX_FILENAME

        histogram_items = "".join(_build_html_list_item(path) for path in files_by_kind.get("histogram_html", []))
        scatter_items = "".join(_build_html_list_item(path) for path in files_by_kind.get("scatter_html", []))
        correlation_items = "".join(_build_html_list_item(path) for path in files_by_kind.get("correlation_html", []))

        first_histogram = next(iter(files_by_kind.get("histogram_html", [])), "")
        first_scatter = next(iter(files_by_kind.get("scatter_html", [])), "")
        first_correlation = next(iter(files_by_kind.get("correlation_html", [])), "")

        dashboard_html = f"""<!DOCTYPE html>
<html lang=\"pt-BR\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Exploração do Dataset</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f3efe6;
            --panel: #fffdf8;
            --ink: #1d2a33;
            --accent: #0f766e;
            --border: #d9d0c0;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: linear-gradient(180deg, #f8f4ea 0%, #efe7d7 100%); color: var(--ink); }}
        header {{ padding: 32px 40px 16px; }}
        h1 {{ margin: 0 0 8px; font-size: 2rem; }}
        p {{ margin: 0; max-width: 900px; line-height: 1.5; }}
        main {{ padding: 8px 24px 40px; display: grid; gap: 20px; }}
        section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 20px; box-shadow: 0 10px 30px rgba(29, 42, 51, 0.08); }}
        h2 {{ margin-top: 0; font-size: 1.2rem; }}
        ul {{ margin: 0 0 16px; padding-left: 18px; }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        iframe {{ width: 100%; height: 720px; border: 1px solid var(--border); border-radius: 12px; background: #fff; }}
    </style>
</head>
<body>
    <header>
        <h1>Exploração do dataset</h1>
        <p>Gráficos gerados a partir do pipeline para o target <strong>{target_column}</strong>. Cada seção abaixo reúne os artefatos HTML e exibe uma prévia navegável.</p>
    </header>
    <main>
        <section>
            <h2>Histogramas</h2>
            <ul>{histogram_items or '<li>Nenhum histograma HTML foi gerado.</li>'}</ul>
            {f'<iframe src="{Path(first_histogram).name}" title="Histogramas"></iframe>' if first_histogram else ''}
        </section>
        <section>
            <h2>Dispersão por target</h2>
            <ul>{scatter_items or '<li>Nenhum gráfico de dispersão HTML foi gerado.</li>'}</ul>
            {f'<iframe src="{Path(first_scatter).name}" title="Dispersão por target"></iframe>' if first_scatter else ''}
        </section>
        <section>
            <h2>Matriz de correlação</h2>
            <ul>{correlation_items or '<li>Nenhuma matriz de correlação HTML foi gerada.</li>'}</ul>
            {f'<iframe src="{Path(first_correlation).name}" title="Matriz de correlação"></iframe>' if first_correlation else ''}
        </section>
    </main>
</body>
</html>
"""

        dashboard_path.write_text(dashboard_html, encoding="utf-8")
        return dashboard_path


def generate_exploration_artifacts(
    data_frame: pd.DataFrame,
    profile: DatasetProfile,
    target_column: str,
    output_dir: str | Path | None = None,
) -> ExplorationArtifacts | None:
    exploration_dir = get_exploration_dir(output_dir)
    exploration_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = data_frame.drop(columns=[target_column], errors="ignore")
    numeric_columns = get_numeric_feature_columns(feature_frame)
    if not numeric_columns:
        return None

    skipped_columns = [column for column in feature_frame.columns if column not in numeric_columns]
    files_by_kind: dict[str, list[str]] = {
        "histogram_png": [],
        "histogram_html": [],
        "scatter_png": [],
        "scatter_html": [],
        "correlation_png": [],
        "correlation_html": [],
    }

    for page_index, page_columns in enumerate(_partition_columns(numeric_columns, DEFAULT_FEATURES_PER_PAGE), start=1):
        histogram_png_path = exploration_dir / f"histogram_page_{page_index:02d}.png"
        histogram_html_path = exploration_dir / f"histogram_page_{page_index:02d}.html"
        scatter_png_path = exploration_dir / f"scatter_vs_{_sanitize_filename(target_column)}_page_{page_index:02d}.png"
        scatter_html_path = exploration_dir / f"scatter_vs_{_sanitize_filename(target_column)}_page_{page_index:02d}.html"

        _save_histogram_png(data_frame, page_columns, histogram_png_path)
        _save_histogram_html(data_frame, page_columns, histogram_html_path)
        _save_scatter_png(data_frame, target_column, page_columns, scatter_png_path)
        _save_scatter_html(data_frame, target_column, page_columns, scatter_html_path)

        files_by_kind["histogram_png"].append(str(histogram_png_path))
        files_by_kind["histogram_html"].append(str(histogram_html_path))
        files_by_kind["scatter_png"].append(str(scatter_png_path))
        files_by_kind["scatter_html"].append(str(scatter_html_path))

    correlation_matrix = compute_numeric_correlation_matrix(data_frame[numeric_columns])
    if not correlation_matrix.empty and correlation_matrix.shape[1] >= 2:
        correlation_png_path = exploration_dir / "correlation_matrix.png"
        correlation_html_path = exploration_dir / "correlation_matrix.html"
        _save_correlation_png(correlation_matrix, correlation_png_path)
        _save_correlation_html(correlation_matrix, correlation_html_path)
        files_by_kind["correlation_png"].append(str(correlation_png_path))
        files_by_kind["correlation_html"].append(str(correlation_html_path))

    metadata_path = exploration_dir / "exploration_metadata.json"
    dashboard_path = create_exploration_dashboard(
        exploration_dir,
        target_column=target_column,
        files_by_kind=files_by_kind,
    )
    metadata = {
        "dataset_path": str(profile.dataset_path),
        "target_column": target_column,
        "numeric_columns": numeric_columns,
        "skipped_columns": skipped_columns,
        "dashboard_path": str(dashboard_path),
        "files_by_kind": files_by_kind,
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

    return ExplorationArtifacts(
        output_dir=exploration_dir,
        metadata_path=metadata_path,
        dashboard_path=dashboard_path,
        files_by_kind=files_by_kind,
        numeric_columns=numeric_columns,
        skipped_columns=skipped_columns,
    )