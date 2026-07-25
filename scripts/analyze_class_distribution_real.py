"""Análise de distribuição de classes com dados reais (somente pandas, sem sklearn)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FE_DIR = ROOT / "src" / "machine_learning"
sys.path.insert(0, str(FE_DIR))

import feature_engineering as fe  # noqa: E402

OUTPUT_DIR = ROOT / "model" / "exploration"
TARGET = fe.DEFAULT_TARGET_COLUMN
THRESHOLD = fe.MIN_CLASS_PERCENTAGE_THRESHOLD


def main() -> None:
    dataset_path = FE_DIR / fe.DEFAULT_DATASET_FILENAME
    if not dataset_path.exists():
        dataset_path = ROOT / fe.DEFAULT_DATASET_FILENAME
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    frame = pd.read_excel(dataset_path)
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)

    before = fe.compute_class_distribution(frame[TARGET])
    filtered, metadata = fe.filter_classes_by_percentage(frame, TARGET, min_percentage=THRESHOLD)
    after = fe.compute_class_distribution(filtered[TARGET])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before_path = OUTPUT_DIR / f"class_distribution_before_purge_{TARGET}.csv"
    after_path = OUTPUT_DIR / f"class_distribution_after_purge_{TARGET}.csv"
    meta_path = OUTPUT_DIR / "class_distribution_real_metadata.json"

    before.to_csv(before_path, index=False)
    after.to_csv(after_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset: {dataset_path}")
    print(f"Linhas (sem nulos no target): {len(frame)}")
    print(f"Classes antes: {before.shape[0]}")
    print(f"Classes após expurgo ({THRESHOLD}%): {after.shape[0]}")
    print(f"Linhas retidas: {metadata['retained_rows']} ({metadata['retained_rows_percentage']}%)")
    print("\nClasses retidas:")
    print(after.to_string(index=False))
    print(f"\nArtefatos:\n  {before_path}\n  {after_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
