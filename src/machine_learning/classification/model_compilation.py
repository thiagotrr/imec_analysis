"""Compilação de modelos treinados em artefatos `.pkl` consumíveis pela API (Task 006, §1).

Contexto: até a Task 05, `run_classification_workflow` treinava CatBoost/XGBoost
e persistia apenas MÉTRICAS (`classification_summary.csv`/`classification_details.json`)
— o modelo treinado em si era descartado ao final da execução. Isso é suficiente
para avaliação/comparação de algoritmos, mas inviabiliza servir o modelo via API
sem re-treinar a cada chamada (custoso e não-determinístico entre chamadas).

Este módulo resolve isso persistindo o(s) modelo(s) treinado(s) — via ``pickle``,
o mesmo mecanismo já usado por ``data_preparation.save_preprocessing_artifacts``
para o ``preprocessing_pipeline.pkl`` (preferimos manter um único padrão de
serialização no projeto; ``joblib`` está disponível no ambiente mas não traz
benefício relevante para os tamanhos de modelo aqui envolvidos) — acompanhado de
um `.json` de metadados (mesmo stem) com tudo que a camada de inferência (task
futura, ver docs/task006_proximos_passos.md §4) precisa para usar o `.pkl` com
segurança: algoritmo, resampling, colunas de entrada esperadas, classes do
target, métricas de validação e referência ao `class_weight_registry.json`
vigente no momento do treino.

Convenção de nomes (``model/compiled/``):
- ``{algorithm}_{resampling_or_none}.pkl`` + ``.json`` — uma combinação treinada
  (ex.: ``xgboost_smote.pkl``, ``catboost_none.pkl``).
- ``champion.pkl`` + ``.json`` — a combinação com melhor ``f1_macro`` entre as
  compiladas nesta chamada (métrica de referência do projeto, ver
  docs/task05_evolucao_pipeline_modelos_v3.md — F1 macro é o que melhor reflete
  o desempenho em um problema multiclasse fortemente desbalanceado). Como o
  cenário padrão da Task 006 só treina XGBoost+SMOTE (ver
  ``classification.models.DEFAULT_CLASSIFIER_ORDER`` e
  ``ClassificationConfig.resampling_strategies``), na prática o campeão
  coincide com essa única combinação — mas a lógica de seleção abaixo é
  genérica e funciona com qualquer conjunto de combinações informado pelo
  caller (ex.: reativando CatBoost/ADASYN via config).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ..data_preparation import get_model_dir
from .metrics import format_classification_metrics

if TYPE_CHECKING:  # evita import circular: workflow.py importa este módulo.
    from .workflow import ClassificationResult


DEFAULT_COMPILED_DIRNAME = "compiled"
CHAMPION_STEM = "champion"
RESAMPLING_LABEL_NONE = "none"
"""Rótulo textual usado no nome do arquivo/metadado quando `resampling is None`
(mesmo valor usado em `workflow.RESAMPLING_LABEL_NONE`; duplicado aqui — em vez
de importado de `workflow` — justamente para não criar a dependência circular
`workflow -> model_compilation -> workflow`)."""


def get_compiled_model_dir(output_dir: str | Path | None = None) -> Path:
    """Retorna (e garante que existe) o diretório `model/compiled/`."""
    compiled_dir = get_model_dir(output_dir) / DEFAULT_COMPILED_DIRNAME
    compiled_dir.mkdir(parents=True, exist_ok=True)
    return compiled_dir


def _resampling_label(resampling: str | None) -> str:
    return resampling or RESAMPLING_LABEL_NONE


def _artifact_stem(algorithm: str, resampling: str | None) -> str:
    return f"{algorithm}_{_resampling_label(resampling)}"


def compile_model_artifact(
    trained_model: object,
    algorithm: str,
    resampling: str | None,
    feature_columns: Sequence[str],
    target_classes: Sequence[object],
    metrics: dict[str, object],
    tier_thresholds: dict[str, float] | None = None,
    class_weight_registry_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    stem: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    """Persiste ``trained_model`` em ``model/compiled/{stem}.pkl`` + metadados ``.json``.

    ``stem`` por padrão é ``"{algorithm}_{resampling_or_none}"`` (ver
    ``_artifact_stem``); é sobrescrito para ``"champion"`` quando chamado por
    ``compile_classification_results`` para persistir o modelo campeão.

    Os metadados obrigatórios (§1.2 do plano) são: ``algorithm``, ``resampling``,
    ``trained_at``, ``feature_columns``, ``target_classes``, ``metrics``,
    ``tier_thresholds`` e ``class_weight_registry_path``. Retorna o ``Path``
    do `.pkl` gerado (o `.json` correspondente é ``pickle_path.with_suffix(".json")``).
    """
    compiled_dir = get_compiled_model_dir(output_dir)
    resolved_stem = stem or _artifact_stem(algorithm, resampling)
    pickle_path = compiled_dir / f"{resolved_stem}.pkl"
    metadata_path = pickle_path.with_suffix(".json")

    with pickle_path.open("wb") as file_obj:
        pickle.dump(trained_model, file_obj)

    metadata: dict[str, object] = {
        "algorithm": algorithm,
        "resampling": _resampling_label(resampling),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": list(feature_columns),
        "target_classes": [str(label) for label in target_classes],
        "metrics": dict(metrics),
        "tier_thresholds": dict(tier_thresholds) if tier_thresholds else None,
        "class_weight_registry_path": str(class_weight_registry_path) if class_weight_registry_path else None,
        "pickle_path": str(pickle_path),
        **(extra_metadata or {}),
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False, default=str)

    return pickle_path


@dataclass(frozen=True)
class CompiledModelArtifact:
    algorithm: str
    resampling: str | None
    pickle_path: Path
    metadata_path: Path
    metadata: dict[str, object]


@dataclass(frozen=True)
class ModelCompilationSummary:
    compiled: list[CompiledModelArtifact]
    champion: CompiledModelArtifact | None
    """``None`` quando ``compiled`` está vazio (nenhum resultado com
    ``trained_model`` disponível para compilar — ver ``ClassificationResult.trained_model``)."""


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    with metadata_path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def compile_classification_results(
    results: Sequence["ClassificationResult"],
    feature_columns: Sequence[str],
    target_classes: Sequence[object],
    tier_thresholds: dict[str, float] | None = None,
    class_weight_registry_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ModelCompilationSummary:
    """Compila cada ``ClassificationResult`` com modelo treinado disponível e eleito o campeão.

    Para cada resultado com ``trained_model is not None``, persiste
    ``{algorithm}_{resampling}.pkl`` + `.json` (via ``compile_model_artifact``).
    Ao final, seleciona a combinação com maior ``metrics["f1"]`` (F1 macro em
    problemas multiclasse — ver ``metrics.compute_classification_metrics``) e
    persiste uma cópia adicional como ``champion.pkl``/``champion.json``,
    anotada com o campo extra ``champion_of`` indicando de qual combinação
    ela veio.

    Resultados sem ``trained_model`` (ex.: quando o caller monta
    ``ClassificationResult`` manualmente, sem passar o modelo) são ignorados
    silenciosamente — não há o que compilar para eles.
    """
    trainable_results = [result for result in results if getattr(result, "trained_model", None) is not None]
    if not trainable_results:
        return ModelCompilationSummary(compiled=[], champion=None)

    compiled: list[CompiledModelArtifact] = []
    for result in trainable_results:
        pickle_path = compile_model_artifact(
            trained_model=result.trained_model,
            algorithm=result.name,
            resampling=result.resampling,
            feature_columns=feature_columns,
            target_classes=target_classes,
            metrics=format_classification_metrics(result.metrics),
            tier_thresholds=tier_thresholds,
            class_weight_registry_path=class_weight_registry_path,
            output_dir=output_dir,
        )
        metadata_path = pickle_path.with_suffix(".json")
        compiled.append(
            CompiledModelArtifact(
                algorithm=result.name,
                resampling=result.resampling,
                pickle_path=pickle_path,
                metadata_path=metadata_path,
                metadata=_load_metadata(metadata_path),
            )
        )

    champion_index = max(
        range(len(compiled)),
        key=lambda index: compiled[index].metadata.get("metrics", {}).get("f1") or 0.0,
    )
    champion_result = trainable_results[champion_index]
    champion_source = compiled[champion_index]

    champion_pickle_path = compile_model_artifact(
        trained_model=champion_result.trained_model,
        algorithm=champion_result.name,
        resampling=champion_result.resampling,
        feature_columns=feature_columns,
        target_classes=target_classes,
        metrics=format_classification_metrics(champion_result.metrics),
        tier_thresholds=tier_thresholds,
        class_weight_registry_path=class_weight_registry_path,
        output_dir=output_dir,
        stem=CHAMPION_STEM,
        extra_metadata={
            "champion_of": {
                "algorithm": champion_source.algorithm,
                "resampling": _resampling_label(champion_source.resampling),
            }
        },
    )
    champion_metadata_path = champion_pickle_path.with_suffix(".json")
    champion = CompiledModelArtifact(
        algorithm=champion_result.name,
        resampling=champion_result.resampling,
        pickle_path=champion_pickle_path,
        metadata_path=champion_metadata_path,
        metadata=_load_metadata(champion_metadata_path),
    )
    return ModelCompilationSummary(compiled=compiled, champion=champion)
