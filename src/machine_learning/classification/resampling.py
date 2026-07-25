from __future__ import annotations

from sklearn.base import BaseEstimator

try:
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover
    ADASYN = None
    SMOTE = None
    ImbPipeline = None


RESAMPLING_STRATEGIES = ("smote", "adasyn")
"""Técnicas de tratamento de desbalanceamento suportadas (biblioteca ``imbalanced-learn``).

Aplicadas exclusivamente ao conjunto de treino (nunca ao teste): quando o
modelo é encapsulado por ``wrap_with_resampling``, o resampling roda apenas
durante ``.fit()`` (dentro do Pipeline do imbalanced-learn); chamadas de
``.predict()``/``.predict_proba()`` passam direto para o classificador, sem
tocar nos dados de teste/validação.
"""


def build_sampler(strategy: str, random_state: int = 42):
    """Cria o sampler do imbalanced-learn correspondente a ``strategy`` (``"smote"`` ou ``"adasyn"``)."""
    if SMOTE is None or ADASYN is None:
        raise ImportError(
            "A biblioteca 'imbalanced-learn' não está instalada. "
            "Instale com 'pip install imbalanced-learn' para usar SMOTE/ADASYN."
        )
    normalized = strategy.strip().lower()
    if normalized == "smote":
        return SMOTE(random_state=random_state)
    if normalized == "adasyn":
        return ADASYN(random_state=random_state)
    raise ValueError(
        f"Estratégia de resampling desconhecida: {strategy!r}. "
        f"Use um valor em {RESAMPLING_STRATEGIES} ou None para desabilitar."
    )


def wrap_with_resampling(
    estimator: BaseEstimator,
    strategy: str | None,
    random_state: int = 42,
) -> BaseEstimator:
    """Encapsula ``estimator`` em um Pipeline do imbalanced-learn que aplica ``strategy`` apenas no treino.

    Se ``strategy`` for ``None``, retorna o próprio ``estimator`` sem alterações
    (mantém disponível o comportamento "sem resampling" para comparação).
    """
    if strategy is None:
        return estimator
    if ImbPipeline is None:
        raise ImportError(
            "A biblioteca 'imbalanced-learn' não está instalada. "
            "Instale com 'pip install imbalanced-learn' para usar SMOTE/ADASYN."
        )
    sampler = build_sampler(strategy, random_state=random_state)
    return ImbPipeline(steps=[("resample", sampler), ("model", estimator)])
