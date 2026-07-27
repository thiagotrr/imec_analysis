from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


DEFAULT_CLASSIFIER_ORDER = ("xgboost",)
"""Algoritmo(s) executado(s) por padrão no pipeline de classificação.

A partir da Task 006 (ver docs/task006_proximos_passos.md e, especialmente,
docs/task05_evolucao_pipeline_modelos_v3.md), o cenário padrão passa a ser
SOMENTE XGBoost. Racional: dos 6 combos testados com dados reais na v3
(CatBoost/XGBoost × nenhum/SMOTE/ADASYN), XGBoost superou o CatBoost em
accuracy/precision/F1 macro em praticamente todos os cenários — o melhor
resultado geral foi XGBoost+SMOTE (F1 macro 61,14%), contra 56,81% do
CatBoost+SMOTE. CatBoost só teve ROC-AUC levemente superior (~98,5% vs.
~98,1-98,7%), o que não compensa a perda de F1 macro, a métrica de
referência do projeto dado o forte desbalanceamento multiclasse do
problema (ver "Comparativo macro" em docs/task05_evolucao_pipeline_modelos_v3.md).

CatBoost (``"catboost"``) permanece **totalmente implementado e mantido**
em ``build_classifier_registry``/``run_classifier_training`` — não foi
removido do código, apenas descontinuado do fluxo *padrão*, seguindo
exatamente o mesmo padrão já usado para k-NN e SVM (ver docstring abaixo):
está disponível para uso pontual, comparativo ou de auditoria a qualquer
momento. Para reativá-lo, informe ``algorithm_order=("catboost", "xgboost")``
(ou a combinação desejada, ex.: incluindo k-NN/SVM) em ``ClassificationConfig``.

k-NN (``"knn"``) e SVM (``"svm"``) seguem o mesmo padrão desde a v2: também
permanecem implementados e disponíveis em ``build_classifier_registry`` para
uso pontual/experimental, descontinuados do fluxo padrão por desempenho
inferior e maior custo de treino/inferência em relação a CatBoost e XGBoost
(ver docs/task05_evolucao_pipeline_modelos.md). Reative-os do mesmo jeito:
``algorithm_order=("catboost", "xgboost", "knn", "svm")`` (ou a combinação
desejada).
"""


DEFAULT_CATBOOST_PARAMS: dict[str, object] = {
    "depth": 6,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "auto_class_weights": "Balanced",
}
"""Hiperparâmetros padrão do CatBoost, ajustados para favorecer recall/F1 das
classes minoritárias em problemas multiclasse muito desbalanceados.

``auto_class_weights="Balanced"`` pondera a função de perda pelo inverso da
frequência de cada classe (equivalente ao ``class_weight="balanced"`` do
scikit-learn), o que costuma elevar recall/F1 macro em detrimento de um
pequeno recuo na accuracy global — troca desejável quando o objetivo é
melhorar a classificação das classes raras. ``depth``/``n_estimators``/
``learning_rate``/``l2_leaf_reg`` foram escolhidos como um ponto de partida
razoável (mais árvores e taxa de aprendizado menor, com regularização L2
moderada para conter overfitting nas classes raras).

IMPORTANTE: estes valores são uma proposta fundamentada em boas práticas para
dados tabulares desbalanceados; eles NÃO foram validados empiricamente neste
projeto porque o dataset real (`resultado_laudo_afericao.xlsx` /
`prepared_training_dataset.csv`) não estava disponível no ambiente em que
foram implementados. Use `hyperparameter_search.run_hyperparameter_search`
(ou ative `ClassificationConfig.enable_hyperparameter_search=True`) para
buscar valores calibrados com dados reais assim que estiverem disponíveis.
"""

DEFAULT_XGBOOST_PARAMS: dict[str, object] = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
"""Hiperparâmetros padrão do XGBoost (mesmo racional do CatBoost acima).

O XGBoost não possui um equivalente direto a ``class_weight`` para problemas
multiclasse (``scale_pos_weight`` só se aplica ao caso binário), então o
balanceamento de classes é aplicado via ``sample_weight`` calculado com
``sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)`` no
momento do treino (ver ``run_classifier_training``). ``min_child_weight`` e
``subsample`` foram elevados/reduzidos moderadamente para reduzir overfitting
em classes raras. Mesma ressalva de validação empírica pendente do CatBoost
se aplica aqui.
"""


@dataclass(frozen=True)
class ClassifierTrainingResult:
    name: str
    model: BaseEstimator
    y_pred: np.ndarray
    y_score: np.ndarray | None
    classes: np.ndarray | None


def build_classifier_registry(
    random_state: int = 42,
    n_classes: int = 2,
    catboost_params: dict[str, object] | None = None,
    xgboost_params: dict[str, object] | None = None,
) -> dict[str, BaseEstimator]:
    registry: dict[str, BaseEstimator] = {
        "knn": KNeighborsClassifier(n_neighbors=7),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state),
    }
    if CatBoostClassifier is not None:
        resolved_catboost_params = {**DEFAULT_CATBOOST_PARAMS, **(catboost_params or {})}
        registry["catboost"] = CatBoostClassifier(
            verbose=False,
            random_seed=random_state,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            # Evita gravar `catboost_info/` (snapshots/logs de treino): além de
            # sujar o repositório a cada execução, gravações concorrentes desse
            # diretório causaram `UnicodeDecodeError` intermitente no Windows
            # quando o CatBoost é treinado várias vezes em sequência (ex.:
            # múltiplos cenários de resampling/CV na mesma execução).
            allow_writing_files=False,
            **resolved_catboost_params,
        )
    if XGBClassifier is not None:
        xgb_params: dict[str, object] = {
            "random_state": random_state,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            **DEFAULT_XGBOOST_PARAMS,
            **(xgboost_params or {}),
        }
        if n_classes > 2:
            xgb_params.update({"objective": "multi:softprob", "num_class": n_classes})
        else:
            xgb_params.update({"objective": "binary:logistic"})
        registry["xgboost"] = XGBClassifier(**xgb_params)
    return registry


def _fit_supports_sample_weight(model: BaseEstimator) -> bool:
    try:
        fit_target = model.fit
        if hasattr(model, "steps"):  # sklearn/imblearn Pipeline: checa o passo final
            fit_target = model.steps[-1][1].fit
        parameters = inspect.signature(fit_target).parameters
        return "sample_weight" in parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def run_classifier_training(
    name: str,
    model: BaseEstimator,
    x_train,
    y_train,
    x_test,
    sample_weight: np.ndarray | None = None,
) -> ClassifierTrainingResult:
    fit_kwargs: dict[str, object] = {}
    if sample_weight is not None:
        if hasattr(model, "steps"):  # Pipeline (ex.: com resampling): repassa para o passo final
            final_step_name = model.steps[-1][0]
            if _fit_supports_sample_weight(model):
                fit_kwargs[f"{final_step_name}__sample_weight"] = sample_weight
        elif _fit_supports_sample_weight(model):
            fit_kwargs["sample_weight"] = sample_weight

    trained_model = model.fit(x_train, y_train, **fit_kwargs)
    y_pred = np.asarray(trained_model.predict(x_test)).ravel()
    y_score = trained_model.predict_proba(x_test) if hasattr(trained_model, "predict_proba") else None
    classes = np.asarray(getattr(trained_model, "classes_", [])) if hasattr(trained_model, "classes_") else None
    return ClassifierTrainingResult(name=name, model=trained_model, y_pred=y_pred, y_score=y_score, classes=classes)
