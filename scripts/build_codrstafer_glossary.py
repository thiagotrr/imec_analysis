"""Gera `model/codrstafer_glossary.json` a partir de `docs/CadastroResultadoDeAfericao.xlsx`.

Fonte oficial de descrições operacionais de CODRSTAFER (Task 009), substituindo o
scaffold anterior (label/description=null). Considera também `SITRSTAFER`
(A=Aprovado / R=Reprovado), usada tanto no prompt da revisão LLM quanto exposta
como novo campo `situacao_afericao` na response da API.

Idempotente: pode ser reexecutado sempre que a planilha for atualizada.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from log import get_log  # noqa: E402

log = get_log()

DEFAULT_INPUT_PATH = ROOT / "docs" / "CadastroResultadoDeAfericao.xlsx"
DEFAULT_OUTPUT_PATH = ROOT / "model" / "codrstafer_glossary.json"
DEFAULT_CLASS_WEIGHT_REGISTRY_PATH = ROOT / "model" / "class_weight_registry.json"

REQUIRED_COLUMNS = ("CODRSTAFER", "DSCRSTAFER", "SITRSTAFER")

_SITUACAO_LABELS = {"A": "Aprovado", "R": "Reprovado"}

GLOSSARY_VERSION = "1.0.0"
GLOSSARY_DESCRIPTION = (
    "Glossário operacional das classes de CODRSTAFER, gerado a partir do cadastro oficial "
    "docs/CadastroResultadoDeAfericao.xlsx (colunas CODRSTAFER/DSCRSTAFER/SITRSTAFER). "
    "Classes com status='confirmado' têm descrição e situação (Aprovado/Reprovado) "
    "homologadas pela área; classes com status='pendente' constam no cadastro mas sem "
    "descrição registrada (DSCRSTAFER vazia/placeholder) — o LLM não deve inventar significado."
)


def _load_class_tiers(path: Path) -> dict[str, str]:
    """Lê `tier` por classe de `class_weight_registry.json` (fail-soft)."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Falha ao ler %s para enriquecer tier no glossário.", path)
        return {}
    tiers: dict[str, str] = {}
    for entry in payload.get("classes") or []:
        classe = entry.get("class")
        tier = entry.get("tier")
        if classe is not None and tier is not None:
            tiers[str(classe)] = str(tier)
    return tiers


def _situacao_label(codigo: str | None) -> str | None:
    if codigo is None:
        return None
    return _SITUACAO_LABELS.get(codigo)


def build_glossary_classes(
    frame: pd.DataFrame,
    *,
    class_tiers: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Monta o bloco `classes` do glossário a partir do DataFrame do cadastro."""
    class_tiers = class_tiers or {}
    classes: dict[str, dict[str, Any]] = {}

    before = len(frame)
    frame = frame.dropna(subset=["CODRSTAFER"])
    if len(frame) != before:
        log.warning("Descartadas %d linha(s) com CODRSTAFER nulo.", before - len(frame))

    duplicated = frame["CODRSTAFER"].duplicated(keep="first")
    if duplicated.any():
        dup_codes = sorted(set(frame.loc[duplicated, "CODRSTAFER"].astype(str)))
        log.warning("CODRSTAFER duplicado no cadastro (mantida a 1ª ocorrência): %s", dup_codes)
        frame = frame.loc[~duplicated]

    for _, row in frame.iterrows():
        code = str(row["CODRSTAFER"]).strip()
        raw_description = str(row["DSCRSTAFER"]).strip()
        # Linhas placeholder: DSCRSTAFER igual ao próprio código (sem descrição real cadastrada).
        is_placeholder = not raw_description or raw_description == code
        description = None if is_placeholder else raw_description

        situacao_raw = row.get("SITRSTAFER")
        situacao_codigo = str(situacao_raw).strip().upper() if pd.notna(situacao_raw) else None
        if situacao_codigo not in (None, "A", "R"):
            log.warning("SITRSTAFER inesperado para CODRSTAFER=%s: %r", code, situacao_raw)
            situacao_codigo = None

        classes[code] = {
            "label": description,
            "description": description,
            "situacao_codigo": situacao_codigo,
            "situacao_label": _situacao_label(situacao_codigo),
            "status": "pendente" if description is None else "confirmado",
            "tier": class_tiers.get(code),
        }

    return classes


def build_glossary_payload(
    input_path: Path = DEFAULT_INPUT_PATH,
    *,
    class_weight_registry_path: Path = DEFAULT_CLASS_WEIGHT_REGISTRY_PATH,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Cadastro não encontrado: {input_path}")

    frame = pd.read_excel(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes em {input_path}: {', '.join(missing_columns)} "
            f"(colunas encontradas: {list(frame.columns)})"
        )

    class_tiers = _load_class_tiers(class_weight_registry_path)
    classes = build_glossary_classes(frame, class_tiers=class_tiers)

    return {
        "version": GLOSSARY_VERSION,
        "target_column": "CODRSTAFER",
        "status": "confirmado",
        "description": GLOSSARY_DESCRIPTION,
        "source_file": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "classes": classes,
    }


def main() -> None:
    payload = build_glossary_payload()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )

    classes = payload["classes"]
    confirmadas = sum(1 for entry in classes.values() if entry["status"] == "confirmado")
    aprovadas = sum(1 for entry in classes.values() if entry["situacao_codigo"] == "A")
    reprovadas = sum(1 for entry in classes.values() if entry["situacao_codigo"] == "R")

    print(f"Cadastro: {DEFAULT_INPUT_PATH}")
    print(f"Classes catalogadas: {len(classes)}")
    print(f"  confirmadas (com descrição): {confirmadas}")
    print(f"  pendentes (sem descrição no cadastro): {len(classes) - confirmadas}")
    print(f"  situação Aprovado (A): {aprovadas} | Reprovado (R): {reprovadas}")
    print(f"Glossário gravado em: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
