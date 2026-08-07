"""Carregamento do glossário operacional de CODRSTAFER (Task 008)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from machine_learning.data_preparation import get_model_dir

DEFAULT_CODRSTAFER_GLOSSARY_FILENAME = "codrstafer_glossary.json"


@dataclass(frozen=True)
class GlossaryEntry:
    code: str
    label: str | None
    description: str | None
    status: str = "pending"
    situacao_codigo: str | None = None
    situacao_label: str | None = None

    def prompt_text(self) -> str:
        label = self.label or "(rótulo pendente)"
        description = self.description or "(descrição operacional pendente no glossário)"
        situacao = self.situacao_label or "(situação não informada)"
        return f"CODRSTAFER {self.code}: {label} — {description} (Situação: {situacao})"


@dataclass(frozen=True)
class CodrstaferGlossary:
    version: str
    status: str
    entries: dict[str, GlossaryEntry]

    def get(self, code: str) -> GlossaryEntry | None:
        return self.entries.get(str(code))

    def prompt_block_for(self, codes: list[str]) -> str:
        lines: list[str] = []
        for code in codes:
            entry = self.get(code)
            if entry is None:
                lines.append(
                    f"CODRSTAFER {code}: (ausente no glossário) — "
                    "não inventar significado operacional."
                )
            else:
                lines.append(entry.prompt_text())
        return "\n".join(lines)


def empty_glossary() -> CodrstaferGlossary:
    return CodrstaferGlossary(version="0.0.0", status="empty", entries={})


def _parse_entry(code: str, raw: Mapping[str, Any] | None) -> GlossaryEntry:
    data = raw or {}
    return GlossaryEntry(
        code=str(code),
        label=data.get("label"),
        description=data.get("description"),
        status=str(data.get("status") or "pending"),
        situacao_codigo=data.get("situacao_codigo"),
        situacao_label=data.get("situacao_label"),
    )


def load_codrstafer_glossary(
    path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> CodrstaferGlossary:
    """Carrega ``model/codrstafer_glossary.json``; fail-soft se ausente/inválido."""
    resolved = Path(path) if path is not None else get_model_dir(output_dir) / DEFAULT_CODRSTAFER_GLOSSARY_FILENAME
    if not resolved.exists():
        return empty_glossary()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_glossary()

    raw_classes = payload.get("classes") or {}
    entries: dict[str, GlossaryEntry] = {}
    if isinstance(raw_classes, Mapping):
        for code, raw_entry in raw_classes.items():
            if isinstance(raw_entry, Mapping):
                entries[str(code)] = _parse_entry(str(code), raw_entry)
            else:
                entries[str(code)] = _parse_entry(str(code), None)

    return CodrstaferGlossary(
        version=str(payload.get("version") or "0.0.0"),
        status=str(payload.get("status") or "unknown"),
        entries=entries,
    )
