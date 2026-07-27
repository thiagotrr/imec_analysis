from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..data_preparation import DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME, get_model_dir
from ..feature_engineering import (
    CLASS_TIER_DISCARD_LABEL,
    CLASS_TIER_THRESHOLDS,
    DEFAULT_TARGET_COLUMN,
    MIN_CLASS_PERCENTAGE_THRESHOLD,
    compute_class_distribution,
    filter_classes_by_percentage,
    qualify_class_distribution,
)
from .cross_validation import DEFAULT_CV_FOLDS, run_stratified_cross_validation
from .hyperparameter_search import (
    DEFAULT_SEARCH_CV_FOLDS,
    DEFAULT_SEARCH_ITERATIONS,
    DEFAULT_SEARCH_SCORING,
    HYPERPARAMETER_SEARCH_SUPPORTED_ALGORITHMS,
    HyperparameterSearchResult,
    run_hyperparameter_search,
)
from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_per_class_metrics,
    format_classification_metrics,
)
from .model_compilation import ModelCompilationSummary, compile_classification_results
from .models import DEFAULT_CLASSIFIER_ORDER, build_classifier_registry, run_classifier_training
from .resampling import wrap_with_resampling


DEFAULT_CLASSIFICATION_DIRNAME = "classification"
DEFAULT_CLASSIFICATION_SUMMARY_FILENAME = "classification_summary.csv"
DEFAULT_CLASSIFICATION_DETAILS_FILENAME = "classification_details.json"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MIN_CLASS_COUNT = 2
"""Rede de segurança complementar ao expurgo por percentual (`min_class_percentage`):
remove classes residuais com contagem absoluta abaixo deste valor, evitando
falhas no split estratificado (`train_test_split(..., stratify=y)`) mesmo se
`min_class_percentage` for configurado com um valor muito baixo."""

RESAMPLING_LABEL_NONE = "none"


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
    min_class_percentage: float = MIN_CLASS_PERCENTAGE_THRESHOLD
    persist_artifacts: bool = True
    output_dir: str | Path | None = None
    dataset_path: str | Path | None = None

    # Fase 2: tratamento de desbalanceamento (SMOTE/ADASYN) e validação cruzada.
    resampling_strategies: tuple[str | None, ...] = ("smote",)
    """Cenários de resampling a executar e comparar, ex.: ``(None, "smote", "adasyn")``.
    ``None`` representa o cenário "sem resampling".

    A partir da Task 006 (ver docs/task05_evolucao_pipeline_modelos_v3.md), o
    padrão passa a ser SOMENTE SMOTE. Racional: nos 6 combos testados com
    dados reais, SMOTE foi neutro/levemente positivo em ambos os algoritmos
    (XGBoost+SMOTE teve o melhor F1 macro geral, 61,14%, ~1,6pp acima de
    XGBoost sem resampling; CatBoost+SMOTE 56,81% vs. 56,51% sem resampling),
    enquanto ADASYN **piorou** o desempenho em ambos (XGBoost: 59,52% → 57,84%;
    CatBoost: 56,51% → 53,87%) e com efeito inconsistente por classe — por
    isso ADASYN deixa de ser executado por padrão.

    "Nenhum resampling" e ADASYN continuam totalmente implementados e
    acessíveis explicitamente via ``resampling_strategies=(None,)`` /
    ``resampling_strategies=("adasyn",)`` (ou qualquer combinação, ex.:
    ``(None, "smote", "adasyn")`` para reproduzir a comparação lado a lado
    feita na v3 — ver scripts/run_task05_v3.py)."""
    enable_cross_validation: bool = False
    cross_validation_folds: int = DEFAULT_CV_FOLDS

    compile_artifacts: bool = True
    """Controla se a etapa de "compilação" de modelos (persistência dos
    modelos treinados em ``model/compiled/*.pkl`` + metadados ``.json``, ver
    ``classification.model_compilation``) roda ao final do workflow —
    análogo a ``persist_artifacts``, mas para os artefatos de INFERÊNCIA
    (não confundir com os artefatos de preparação salvos por
    ``data_preparation.save_preprocessing_artifacts``).

    Reaproveita os modelos já treinados nesta execução (via
    ``ClassificationResult.trained_model``), sem re-treinar. Como o cenário
    padrão agora só treina XGBoost+SMOTE (ver ``DEFAULT_CLASSIFIER_ORDER`` e
    o default de ``resampling_strategies`` acima), na prática o "campeão"
    (``model/compiled/champion.pkl``) coincide com essa única combinação —
    mas a etapa é genérica e funciona com qualquer conjunto de combinações
    (ex.: incluindo CatBoost) informado via ``algorithm_order``/
    ``resampling_strategies``."""

    # Fase 1: busca de hiperparâmetros (opcional, custosa - ver hyperparameter_search.py).
    enable_hyperparameter_search: bool = False
    hyperparameter_search_scoring: str = DEFAULT_SEARCH_SCORING
    hyperparameter_search_iterations: int = DEFAULT_SEARCH_ITERATIONS
    hyperparameter_search_cv_folds: int = DEFAULT_SEARCH_CV_FOLDS


@dataclass(frozen=True)
class ClassificationResult:
    name: str
    metrics: ClassificationMetrics
    model_name: str
    resampling: str | None = None
    cross_validation: dict[str, object] | None = None
    hyperparameter_search: dict[str, object] | None = None
    per_class: list[dict[str, object]] | None = None
    """Precision/recall/f1/support por classe (rótulo original + camada de
    qualificação A/B/C/D), no conjunto de teste — ver ``metrics.compute_per_class_metrics``
    e ``_decorate_per_class_with_labels_and_tiers``. Permite avaliar o efeito
    do resampling especificamente nas classes das camadas B/C, em vez de
    apenas nas médias macro."""
    trained_model: object = None
    """Referência ao estimador `sklearn`/`imblearn` já treinado (ex.:
    `XGBClassifier` ou o `Pipeline` de resampling que o encapsula).

    Adicionado na Task 006 exclusivamente para permitir a etapa de
    "compilação" (ver `model_compilation.compile_classification_results`)
    reaproveitar o modelo já treinado nesta execução, sem re-treinar.
    Deliberadamente NÃO é incluído em `results_payload`/`consolidated`
    (as estruturas persistidas em `classification_details.json`/
    `classification_summary.csv`, que continuam construídas manualmente
    campo a campo abaixo) — serializar um objeto de modelo em JSON não
    faz sentido e poderia vazar dados binários/grandes no relatório."""


@dataclass(frozen=True)
class ClassificationWorkflowResult:
    results: list[ClassificationResult]
    consolidated: pd.DataFrame
    artifacts: ClassificationArtifacts | None
    split_metadata: dict[str, object]
    hyperparameter_search_results: list[HyperparameterSearchResult]
    compiled_models: ModelCompilationSummary | None = None
    """Resultado da etapa de compilação (ver `ClassificationConfig.compile_artifacts`
    e `model_compilation.compile_classification_results`); `None` quando
    `compile_artifacts=False` ou quando nenhum resultado tinha `trained_model`
    disponível."""


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
    min_class_percentage: float,
):
    """Filtra classes de baixa representatividade e separa treino/teste de forma estratificada.

    O expurgo é feito em duas etapas complementares:
    1. Percentual (`filter_classes_by_percentage`, critério primário): remove
       classes cujo percentual sobre o total de linhas seja menor que
       `min_class_percentage` — ver `feature_engineering.MIN_CLASS_PERCENTAGE_THRESHOLD`.
    2. Contagem absoluta (`min_class_count`, rede de segurança): remove
       eventuais classes residuais com poucas amostras (ex.: < 2), o que
       inviabilizaria o split estratificado.
    """
    if target_column not in data_frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in prepared dataset.")

    full_distribution = qualify_class_distribution(compute_class_distribution(data_frame[target_column]))

    percentage_filtered_frame, percentage_metadata = filter_classes_by_percentage(
        data_frame, target_column, min_percentage=min_class_percentage
    )
    if percentage_filtered_frame.empty:
        raise ValueError(
            "Nenhuma classe do target atingiu o limiar percentual mínimo "
            f"({min_class_percentage}%) definido em min_class_percentage."
        )

    class_counts = percentage_filtered_frame[target_column].value_counts().sort_index()
    rare_class_counts = class_counts[class_counts < min_class_count]
    filtered_frame = percentage_filtered_frame[
        ~percentage_filtered_frame[target_column].isin(rare_class_counts.index)
    ].copy()
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
        "min_class_percentage": min_class_percentage,
        "min_class_count": int(min_class_count),
        "original_rows": int(len(data_frame)),
        "original_class_count": int(full_distribution.shape[0]),
        "rows_after_percentage_filter": int(len(percentage_filtered_frame)),
        "class_count_after_percentage_filter": int(percentage_filtered_frame[target_column].nunique()),
        "removed_classes_by_percentage": percentage_metadata["removed_classes"],
        "training_rows": int(len(filtered_frame)),
        "training_class_count": int(y.nunique()),
        "removed_rows": int(len(data_frame) - len(filtered_frame)),
        "removed_rare_classes_by_count": {str(label): int(count) for label, count in rare_class_counts.items()},
        "class_distribution_before": full_distribution.to_dict(orient="records"),
        "class_distribution_after": qualify_class_distribution(
            compute_class_distribution(filtered_frame[target_column])
        ).to_dict(orient="records"),
        "target_label_mapping": {
            str(label): int(encoded_label)
            for encoded_label, label in enumerate(target_encoder.classes_)
        },
    }
    return (*split, metadata, x, y)


def _print_result(result: ClassificationResult) -> None:
    metrics = result.metrics
    resampling_label = result.resampling or RESAMPLING_LABEL_NONE
    print(f"\n{result.name} (resampling={resampling_label})")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1: {metrics.f1:.4f}")
    print(f"ROC-AUC: {metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "ROC-AUC: n/a")
    print(f"PR-AUC: {metrics.pr_auc:.4f}" if metrics.pr_auc is not None else "PR-AUC: n/a")
    if result.per_class:
        non_dominant = [entry for entry in result.per_class if entry.get("tier") != "A"]
        if non_dominant:
            print("Por classe (camadas B/C - foco do data augmentation):")
            for entry in non_dominant:
                print(
                    f"  classe={entry['original_class']} (camada {entry['tier']}, n={entry['support']}) "
                    f"precision={entry['precision']:.3f} recall={entry['recall']:.3f} f1={entry['f1']:.3f}"
                )
    if result.cross_validation is not None:
        f1_cv = result.cross_validation.get("f1_macro", {})
        recall_cv = result.cross_validation.get("recall_macro", {})
        roc_auc_cv = result.cross_validation.get("roc_auc_macro", {})
        print(
            "StratifiedKFold "
            f"(k={result.cross_validation.get('cv_folds_used')}): "
            f"f1_macro={f1_cv.get('mean'):.4f}±{f1_cv.get('std'):.4f} "
            f"recall_macro={recall_cv.get('mean'):.4f}±{recall_cv.get('std'):.4f} "
            f"roc_auc_macro={roc_auc_cv.get('mean'):.4f}±{roc_auc_cv.get('std'):.4f}"
        )


def _decorate_per_class_with_labels_and_tiers(
    per_class: list[dict[str, object]],
    label_by_encoded: dict[int, str],
    tier_by_label: dict[str, str],
) -> list[dict[str, object]]:
    decorated = []
    for entry in per_class:
        original_label = label_by_encoded.get(int(entry["class"]))
        decorated.append(
            {
                **entry,
                "original_class": original_label,
                "tier": tier_by_label.get(original_label) if original_label is not None else None,
            }
        )
    # Camadas mais raras (piores candidatas a augmentation) primeiro, para
    # facilitar a leitura de quem mais precisa de atenção no relatório.
    tier_order = {"A": 0, "B": 1, "C": 2, CLASS_TIER_DISCARD_LABEL: 3}
    decorated.sort(key=lambda item: tier_order.get(item["tier"], 99))
    return decorated


def run_classification_workflow(
    config: ClassificationConfig | None = None,
    output_dir: str | Path | None = None,
) -> ClassificationWorkflowResult:
    resolved_config = config or ClassificationConfig(output_dir=output_dir)
    prepared_dataset = _load_prepared_dataset(resolved_config.dataset_path, resolved_config.output_dir)
    x_train, x_test, y_train, y_test, split_metadata, x_full, y_full = _split_dataset(
        prepared_dataset,
        resolved_config.target_column,
        resolved_config.test_size,
        resolved_config.random_state,
        resolved_config.min_class_count,
        resolved_config.min_class_percentage,
    )
    n_classes = int(pd.Series(y_train).nunique())
    if split_metadata["removed_rows"]:
        print(
            "Classes de baixa representatividade removidas do treino/validação: "
            f"{split_metadata['removed_rows']} linhas; "
            f"{split_metadata['original_class_count']} -> {split_metadata['training_class_count']} classes "
            f"(limiar={split_metadata['min_class_percentage']}%)."
        )

    registry = build_classifier_registry(random_state=resolved_config.random_state, n_classes=n_classes)
    results: list[ClassificationResult] = []
    hyperparameter_search_results: list[HyperparameterSearchResult] = []

    label_by_encoded = {
        encoded_label: original_label
        for original_label, encoded_label in split_metadata["target_label_mapping"].items()
    }
    tier_by_label = {
        entry["class"]: entry["tier"] for entry in split_metadata["class_distribution_after"]
    }

    for algorithm_name in resolved_config.algorithm_order:
        base_model = registry.get(algorithm_name)
        if base_model is None:
            continue

        tuned_model = base_model
        search_summary: dict[str, object] | None = None
        if (
            resolved_config.enable_hyperparameter_search
            and algorithm_name in HYPERPARAMETER_SEARCH_SUPPORTED_ALGORITHMS
        ):
            tuned_model, search_result = run_hyperparameter_search(
                algorithm_name,
                base_model,
                x_train,
                y_train,
                scoring=resolved_config.hyperparameter_search_scoring,
                n_iter=resolved_config.hyperparameter_search_iterations,
                cv_folds=resolved_config.hyperparameter_search_cv_folds,
                random_state=resolved_config.random_state,
            )
            hyperparameter_search_results.append(search_result)
            search_summary = asdict(search_result)
            print(
                f"\nBusca de hiperparâmetros ({algorithm_name}): melhor {search_result.scoring}="
                f"{search_result.best_score:.4f} com params={search_result.best_params}"
            )

        for resampling_strategy in resolved_config.resampling_strategies:
            estimator = wrap_with_resampling(clone(tuned_model), resampling_strategy, resolved_config.random_state)

            # Balanceamento via sample_weight só é aplicado quando NÃO há resampling
            # explícito (SMOTE/ADASYN já reequilibram as classes por si mesmos;
            # combinar as duas técnicas seria redundante e complicaria o cálculo
            # do peso, já que o resampling altera o número de linhas de treino).
            sample_weight = None
            if resampling_strategy is None:
                sample_weight = compute_sample_weight("balanced", y_train)

            training_result = run_classifier_training(
                algorithm_name, estimator, x_train, y_train, x_test, sample_weight=sample_weight
            )
            metrics = compute_classification_metrics(
                y_test,
                training_result.y_pred,
                training_result.y_score,
                classes=training_result.classes,
            )
            per_class = _decorate_per_class_with_labels_and_tiers(
                compute_per_class_metrics(y_test, training_result.y_pred, training_result.classes),
                label_by_encoded,
                tier_by_label,
            )

            cross_validation_summary = None
            if resolved_config.enable_cross_validation:
                cv_estimator = wrap_with_resampling(
                    clone(tuned_model), resampling_strategy, resolved_config.random_state
                )
                cross_validation_summary = run_stratified_cross_validation(
                    cv_estimator,
                    x_full,
                    y_full,
                    cv_folds=resolved_config.cross_validation_folds,
                    random_state=resolved_config.random_state,
                )

            result = ClassificationResult(
                name=algorithm_name,
                metrics=metrics,
                model_name=base_model.__class__.__name__,
                resampling=resampling_strategy,
                cross_validation=cross_validation_summary,
                hyperparameter_search=search_summary,
                per_class=per_class,
                trained_model=training_result.model,
            )
            results.append(result)
            _print_result(result)

    consolidated = pd.DataFrame([
        {
            "algorithm": result.name,
            "model": result.model_name,
            "resampling": result.resampling or RESAMPLING_LABEL_NONE,
            **format_classification_metrics(result.metrics),
        }
        for result in results
    ])

    results_payload = [
        {
            "algorithm": result.name,
            "model": result.model_name,
            "resampling": result.resampling or RESAMPLING_LABEL_NONE,
            **format_classification_metrics(result.metrics),
            "cross_validation": result.cross_validation,
            "hyperparameter_search": result.hyperparameter_search,
            "per_class": result.per_class,
        }
        for result in results
    ]

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
                    "resampling_strategies": [
                        strategy or RESAMPLING_LABEL_NONE for strategy in resolved_config.resampling_strategies
                    ],
                    "cross_validation_enabled": resolved_config.enable_cross_validation,
                    "cross_validation_folds": resolved_config.cross_validation_folds,
                    "hyperparameter_search_enabled": resolved_config.enable_hyperparameter_search,
                    "split": split_metadata,
                    "results": results_payload,
                },
                file_obj,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        artifacts = ClassificationArtifacts(output_dir=classification_dir, summary_path=summary_path, details_path=details_path)

    if not consolidated.empty:
        print("\nConsolidado final")
        print(consolidated.to_string(index=False))

    compiled_models: ModelCompilationSummary | None = None
    if resolved_config.compile_artifacts:
        # Reaproveita os modelos já treinados (`ClassificationResult.trained_model`)
        # nesta mesma execução — a compilação NÃO re-treina nada. `feature_columns`
        # é a mesma para todos os resultados desta chamada (mesmo split de treino).
        class_weight_registry_path = get_model_dir(resolved_config.output_dir) / DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME
        compiled_models = compile_classification_results(
            results=results,
            feature_columns=x_train.columns.tolist(),
            target_classes=[label_by_encoded[index] for index in sorted(label_by_encoded)],
            tier_thresholds=CLASS_TIER_THRESHOLDS,
            class_weight_registry_path=class_weight_registry_path if class_weight_registry_path.exists() else None,
            output_dir=resolved_config.output_dir,
        )
        if compiled_models.champion is not None:
            print(
                "\nModelo campeão compilado: "
                f"{compiled_models.champion.algorithm} (resampling="
                f"{compiled_models.champion.resampling or RESAMPLING_LABEL_NONE}) "
                f"-> {compiled_models.champion.pickle_path}"
            )

    return ClassificationWorkflowResult(
        results=results,
        consolidated=consolidated,
        artifacts=artifacts,
        split_metadata=split_metadata,
        hyperparameter_search_results=hyperparameter_search_results,
        compiled_models=compiled_models,
    )
