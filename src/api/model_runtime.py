"""Carga e cache dos artefatos de inferência (Task 007).

Carrega uma vez no ``lifespan`` da API e guarda em ``app.state.runtime``.
Os services de análise consomem este runtime — não relêem `.pkl` a cada request.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from machine_learning.classification.model_compilation import CHAMPION_STEM, get_compiled_model_dir
from machine_learning.data_preparation import (
    DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME,
    DEFAULT_PIPELINE_FILENAME,
    DEFAULT_TARGET_ENCODER_FILENAME,
    get_model_dir,
)

from .models.inspecao_request import RETAINED_FEATURE_COLUMNS


class ModelRuntimeError(RuntimeError):
    """Falha ao carregar artefatos ou runtime indisponível para inferência."""


@dataclass(frozen=True)
class ClassTierInfo:
    classe: str
    tier: str
    weight: float
    count: int | None = None
    percentage: float | None = None


@dataclass(frozen=True)
class ModelRuntime:
    """Artefatos prontos para inferência, carregados no startup."""

    champion: Any
    preprocessing_pipeline: Any
    target_classes: tuple[str, ...]
    class_lookup: dict[str, ClassTierInfo]
    discard_tier_label: str
    retained_feature_columns: tuple[str, ...]
    champion_metadata: dict[str, Any]
    target_encoder: Any | None = None

    def decode_prediction(self, encoded_label: int) -> str:
        if self.target_encoder is not None:
            decoded = self.target_encoder.inverse_transform([encoded_label])[0]
            return str(decoded)
        try:
            return self.target_classes[int(encoded_label)]
        except (IndexError, ValueError, TypeError) as exc:
            raise ModelRuntimeError(
                f"Índice de classe previsto ({encoded_label}) fora de "
                f"target_classes (n={len(self.target_classes)})."
            ) from exc

    def lookup_class(self, classe: str) -> ClassTierInfo | None:
        return self.class_lookup.get(str(classe))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _build_class_lookup(registry: dict[str, Any]) -> dict[str, ClassTierInfo]:
    lookup: dict[str, ClassTierInfo] = {}
    for entry in registry.get("classes") or []:
        classe = str(entry["class"])
        lookup[classe] = ClassTierInfo(
            classe=classe,
            tier=str(entry.get("tier", "D")),
            weight=float(entry.get("weight", 0.0)),
            count=int(entry["count"]) if entry.get("count") is not None else None,
            percentage=float(entry["percentage"]) if entry.get("percentage") is not None else None,
        )
    return lookup


def load_model_runtime(output_dir: str | Path | None = None) -> ModelRuntime:
    """Carrega champion + pipeline + registry (e target_encoder se existir).

    O ``target_encoder.pkl`` pode estar ausente quando ``CODRSTAFER`` já é
    numérico na preparação — nesse caso a decodificação usa
    ``champion.json → target_classes`` (ordem do ``LabelEncoder`` do treino).
    """
    model_dir = get_model_dir(output_dir)
    compiled_dir = get_compiled_model_dir(output_dir)

    champion_pkl = compiled_dir / f"{CHAMPION_STEM}.pkl"
    champion_json = compiled_dir / f"{CHAMPION_STEM}.json"
    pipeline_path = model_dir / DEFAULT_PIPELINE_FILENAME
    registry_path = model_dir / DEFAULT_CLASS_WEIGHT_REGISTRY_FILENAME
    encoder_path = model_dir / DEFAULT_TARGET_ENCODER_FILENAME

    missing = [
        str(path)
        for path in (champion_pkl, champion_json, pipeline_path, registry_path)
        if not path.exists()
    ]
    if missing:
        raise ModelRuntimeError(
            "Artefatos obrigatórios ausentes para inferência: "
            + "; ".join(missing)
            + ". Rode 'python scripts/compile_models.py' (e o pipeline de preparação) antes."
        )

    try:
        champion = _load_pickle(champion_pkl)
        pipeline = _load_pickle(pipeline_path)
        metadata = _load_json(champion_json)
        registry = _load_json(registry_path)
    except (OSError, pickle.UnpicklingError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ModelRuntimeError(f"Falha ao carregar artefatos de inferência: {exc}") from exc

    target_classes = tuple(str(label) for label in (metadata.get("target_classes") or []))
    if not target_classes:
        raise ModelRuntimeError(
            f"{champion_json} não contém target_classes — recompile com scripts/compile_models.py."
        )

    target_encoder = None
    if encoder_path.exists():
        try:
            target_encoder = _load_pickle(encoder_path)
        except (OSError, pickle.UnpicklingError, TypeError, ValueError) as exc:
            raise ModelRuntimeError(f"Falha ao carregar {encoder_path}: {exc}") from exc

    return ModelRuntime(
        champion=champion,
        preprocessing_pipeline=pipeline,
        target_classes=target_classes,
        class_lookup=_build_class_lookup(registry),
        discard_tier_label=str(registry.get("discard_tier_label") or "D"),
        retained_feature_columns=RETAINED_FEATURE_COLUMNS,
        champion_metadata=metadata,
        target_encoder=target_encoder,
    )
