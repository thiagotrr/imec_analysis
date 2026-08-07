"""Configuração compartilhada dos testes.

O projeto não usa um layout de pacote instalável (`src/` não é um pacote
Python com `__init__.py` na raiz); em vez disso, `src/main.py` é executado
com `src` no `sys.path`, e `machine_learning` é importado como pacote
top-level a partir daí (ver `src/main.py`). Replicamos esse mesmo
mecanismo aqui para os testes importarem `machine_learning` da mesma forma
que o código de produção espera.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Testes automatizados NUNCA devem disparar chamadas reais a provedores LLM,
# mesmo que o ambiente (shell/CI) tenha OPENAI_API_KEY/GEMINI_API_KEY reais
# configuradas para uso manual da API. Força LLM_ENABLED=false e remove as
# chaves antes de qualquer import de código de produção — testes que
# exercitam o fluxo de revisão LLM constroem `LlmSettings(enabled=True, ...)`
# explicitamente com `FakeLlmReviewer`, então não dependem dessas variáveis.
os.environ["LLM_ENABLED"] = "false"
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TARGET_COLUMN = "CODRSTAFER"


def _build_synthetic_dataset(n_rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Gera um dataset sintético com o mesmo "formato" do problema real:

    - Uma coluna alvo multiclasse (`CODRSTAFER`) com poucas classes dominantes
      e várias classes muito raras (imitando o forte desbalanceamento real).
    - Features numéricas com sinal (média deslocada por classe) e uma feature
      categórica correlacionada com a classe, para que o pipeline de
      pré-processamento (scaling/encoding/SVD) e os classificadores tenham
      algo não-trivial para aprender.

    IMPORTANTE: estes dados são inteiramente sintéticos e não representam o
    domínio real de laudos de aferição de medidores. Servem apenas para
    validar que o código executa corretamente de ponta a ponta; métricas
    obtidas a partir deles NÃO devem ser interpretadas como desempenho real do
    modelo (ver docs/task05_evolucao_pipeline_modelos.md).
    """
    rng = np.random.default_rng(random_state)

    # Classes dominantes (ficam acima do limiar de 15% após o expurgo) e
    # classes raras (ficam abaixo e devem ser removidas).
    dominant_classes = {101: 0.45, 102: 0.25, 103: 0.16}
    rare_classes = {104: 0.08, 105: 0.015, 106: 0.015, 107: 0.015, 108: 0.015,
                    109: 0.015, 110: 0.015, 111: 0.015, 112: 0.015, 113: 0.015}
    class_probabilities = {**dominant_classes, **rare_classes}
    labels = list(class_probabilities.keys())
    probabilities = np.array(list(class_probabilities.values()))
    probabilities = probabilities / probabilities.sum()

    target = rng.choice(labels, size=n_rows, p=probabilities)
    class_to_shift = {label: index * 2.5 for index, label in enumerate(sorted(class_probabilities))}

    # Arredondado para poucas casas decimais de propósito: valores contínuos
    # com cardinalidade == número de linhas disparariam a heurística de
    # "coluna tipo ID" em `feature_engineering.build_dataset_profile` e
    # seriam descartados do pipeline, o que não reflete o problema real (cujas
    # colunas numéricas não são identificadores).
    numeric_feature_1 = np.round(
        np.array([class_to_shift[label] for label in target]) + rng.normal(0, 1.5, size=n_rows), 1
    )
    numeric_feature_2 = np.round(
        rng.normal(10, 3, size=n_rows) + np.array([0.5 * class_to_shift[label] for label in target]), 1
    )
    numeric_feature_3 = np.round(rng.exponential(2.0, size=n_rows), 1)
    # Cada classe tem uma categoria "preferida" (derivada do próprio rótulo),
    # com alguma chance de ruído — garante que múltiplas categorias apareçam
    # mesmo dentro de uma única classe (incluindo as classes dominantes que
    # sobrevivem ao expurgo de 15%).
    category_options = np.array(["A", "B", "C", "D"])
    preferred_category = category_options[np.array([label % 4 for label in target])]
    is_noisy = rng.random(n_rows) < 0.3
    categorical_feature = np.where(is_noisy, rng.choice(category_options, size=n_rows), preferred_category)

    return pd.DataFrame(
        {
            "NUMERIC_FEATURE_1": numeric_feature_1,
            "NUMERIC_FEATURE_2": numeric_feature_2,
            "NUMERIC_FEATURE_3": numeric_feature_3,
            "CATEGORICAL_FEATURE": categorical_feature,
            TARGET_COLUMN: target,
        }
    )


@pytest.fixture()
def synthetic_dataset() -> pd.DataFrame:
    return _build_synthetic_dataset()
