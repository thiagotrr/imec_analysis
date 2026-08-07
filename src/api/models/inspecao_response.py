"""Contratos Pydantic de resposta dos endpoints de inspeção de medidor (Task 006/007)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InspecaoLaudoResponse(BaseModel):
    """Resposta dos endpoints de análise de laudo (`/inspecao/laudo_completo`,
    `/inspecao/laudo_sintetico` e cada item de `/inspecao/csv`)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "numero_laudo": "2025006988",
                "classe_prevista": "10",
                "camada": "A",
                "situacao_afericao": "Reprovado",
                "resultado": "Classe 10 (camada A)",
                "resultado_detalhado": (
                    "Classe prevista: 10 (camada A). Camada A: classe dominante no histórico "
                    "de treino (≥15% das amostras) — sinal de maior confiabilidade estatística. "
                    "Peso balanceado da classe no treino: 0.153795. "
                    "Probabilidade da classe prevista (predict_proba): 0.9123."
                ),
                "predict_proba": {"10": 0.9123},
                "revisao_llm": None,
            }
        }
    )

    numero_laudo: str | None = Field(
        default=None,
        description="Identificador do laudo de aferição (`NUMLAUDO`), quando presente no payload de entrada.",
    )
    classe_prevista: str | None = Field(
        default=None,
        description=(
            "Classe prevista pelo modelo para `CODRSTAFER` (rótulo original, decodificado via "
            "`target_encoder.pkl` ou `champion.json → target_classes`)."
        ),
    )
    camada: str | None = Field(
        default=None,
        description=(
            "Camada de qualificação (A/B/C/D) da classe prevista, conforme "
            "`model/class_weight_registry.json`."
        ),
    )
    situacao_afericao: str | None = Field(
        default=None,
        description=(
            "Situação da aferição associada à classe prevista ('Aprovado' ou 'Reprovado'), "
            "derivada de `SITRSTAFER` no cadastro oficial "
            "(`model/codrstafer_glossary.json`, campo `situacao_label`). "
            "Null quando a classe prevista não consta no glossário."
        ),
    )
    resultado: str = Field(
        ...,
        description=(
            "Resultado resumido: 'Classe {código} (camada {A|B|C})' ou "
            "'Revisão manual' quando a classe é D / fora do registry."
        ),
    )
    resultado_detalhado: str = Field(
        ...,
        description=(
            "Texto detalhado da análise (template A–C citando camada/peso e, quando disponível, "
            "a probabilidade da classe prevista)."
        ),
    )
    predict_proba: dict[str, float] | None = Field(
        default=None,
        description=(
            "Distribuição de probabilidades por rótulo original de `CODRSTAFER` "
            "(saída de `champion.predict_proba`), quando o modelo expõe essa API. "
            "Ordenado da maior para a menor probabilidade, com a classe prevista sempre em "
            "primeiro lugar. Quando a probabilidade da classe prevista for ≥ 90%, este campo "
            "traz apenas essa classe (sem alternativas), para evitar ambiguidade."
        ),
    )

    revisao_llm: str | None = Field(
        default=None,
        description=(
            "Revisão em linguagem natural gerada por LLM (pós-processamento opcional). "
            "Não altera `classe_prevista`, `camada` nem `predict_proba`. "
            "Null quando o gate omite a chamada (ex.: camada A por default), "
            "quando LLM está desabilitado/sem chave, no CSV em lote, ou em falha fail-soft."
        ),
    )


class InspecaoLaudoCsvItemResponse(InspecaoLaudoResponse):
    """Item de resposta do upload em lote (`/inspecao/csv`)."""

    numero_linha: int = Field(..., description="Número da linha no CSV recebido (1-based, sem contar o cabeçalho).")


class ModeloInfoAlgoritmoResponse(BaseModel):
    """Metadados de UMA combinação algoritmo+resampling compilada."""

    algorithm: str = Field(..., description="Nome do algoritmo (ex.: 'xgboost', 'catboost').")
    resampling: str = Field(..., description="Estratégia de resampling usada no treino ('none', 'smote' ou 'adasyn').")
    trained_at: str | None = Field(default=None, description="Timestamp ISO-8601 (UTC) do treino que gerou este artefato.")
    metrics: dict[str, float | str | bool | None] = Field(
        default_factory=dict, description="Métricas de validação (accuracy/precision/recall/f1/roc_auc/pr_auc macro)."
    )
    feature_columns_count: int | None = Field(
        default=None, description="Número de colunas de entrada esperadas pelo `preprocessing_pipeline.pkl`."
    )
    target_classes: list[str] | None = Field(
        default=None, description="Rótulos originais de `CODRSTAFER` suportados por este modelo."
    )


class ModeloInfoResponse(BaseModel):
    """Resposta de `GET /inspecao/modelos`: metadados do(s) modelo(s) compilado(s)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "champion": {
                    "algorithm": "xgboost",
                    "resampling": "smote",
                    "trained_at": "2026-07-24T22:37:02+00:00",
                    "metrics": {"accuracy": 0.9344, "f1": 0.6114, "roc_auc": 0.9812},
                    "feature_columns_count": 128,
                    "target_classes": ["1", "10", "165"],
                },
                "compiled_variants": [],
                "tier_thresholds": {"A": 15.0, "B": 1.0, "C": 0.1},
                "class_weight_registry_available": True,
                "message": None,
            }
        }
    )

    champion: ModeloInfoAlgoritmoResponse | None = Field(
        default=None, description="Metadados do modelo campeão (`model/compiled/champion.json`), se já compilado."
    )
    compiled_variants: list[ModeloInfoAlgoritmoResponse] = Field(
        default_factory=list,
        description="Demais combinações algoritmo+resampling compiladas (excluindo o campeão), se houver.",
    )
    tier_thresholds: dict[str, float] | None = Field(
        default=None, description="Limiares de camada A/B/C usados no treino do modelo campeão."
    )
    class_weight_registry_available: bool = Field(
        default=False, description="Indica se `model/class_weight_registry.json` está disponível."
    )
    message: str | None = Field(
        default=None,
        description="Mensagem informativa quando nenhum modelo compilado for encontrado.",
    )
