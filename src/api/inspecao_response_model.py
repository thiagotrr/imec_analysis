"""Contratos Pydantic de resposta dos endpoints de inspeção de medidor (Task 006, §2.4/§3)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InspecaoLaudoResponse(BaseModel):
    """Resposta dos endpoints de análise de laudo (`/inspecao/laudo_completo`,
    `/inspecao/laudo_sintetico` e cada item de `/inspecao/csv`).

    Nesta task NENHUM destes campos é populado por lógica de inferência real
    (ver docs/task006_proximos_passos.md, §4 "Fora do escopo") — o contrato
    de resposta é desenhado agora para já refletir o formato esperado pela
    futura task de "services", mas os handlers atuais são stubs que
    levantam `NotImplementedError` (convertido em HTTP 500, ver
    `inspecao_router.py`)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "numero_laudo": "2025006988",
                "classe_prevista": None,
                "camada": None,
                "resultado": "Não implementado",
                "resultado_detalhado": (
                    "Service de inferência ainda não implementado — ver "
                    "docs/task006_proximos_passos.md, §4."
                ),
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
            "`target_encoder.pkl`). `None` enquanto o service de inferência não estiver implementado."
        ),
    )
    camada: str | None = Field(
        default=None,
        description=(
            "Camada de qualificação (A/B/C/D, ver `feature_engineering.CLASS_TIER_THRESHOLDS`) da "
            "classe prevista, conforme `model/class_weight_registry.json` — insumo para a futura "
            "narrativa de confiabilidade via LLM (ver docs/task006_proximos_passos.md, §2.4/§4)."
        ),
    )
    resultado: str = Field(..., description="Resultado resumido da inspeção (ex.: 'Aprovado', 'Reprovado', 'Revisão manual').")
    resultado_detalhado: str = Field(
        ...,
        description=(
            "Texto detalhado da análise. Campo reservado para a futura narrativa de "
            "confiabilidade/camada gerada via LLM a partir de `class_weight_registry.json` "
            "(ver docs/task006_proximos_passos.md, §2.4 e §4) — nesta task, apenas descreve "
            "por que a inferência real ainda não está disponível."
        ),
    )


class InspecaoLaudoCsvItemResponse(InspecaoLaudoResponse):
    """Item de resposta do upload em lote (`/inspecao/csv`) — estende
    `InspecaoLaudoResponse` com a posição da linha no CSV recebido, para o
    consumidor conseguir relacionar cada resultado à linha de origem."""

    numero_linha: int = Field(..., description="Número da linha no CSV recebido (1-based, sem contar o cabeçalho).")


class ModeloInfoAlgoritmoResponse(BaseModel):
    """Metadados de UMA combinação algoritmo+resampling compilada (espelha o
    `.json` gerado por `classification.model_compilation.compile_model_artifact`)."""

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
        default=None, description="Rótulos originais de `CODRSTAFER` suportados por este modelo (via `target_encoder.pkl`)."
    )


class ModeloInfoResponse(BaseModel):
    """Resposta de `GET /inspecao/modelos`: metadados do(s) modelo(s)
    compilado(s) disponíveis em `model/compiled/` (ver
    `scripts/compile_models.py`) — útil para consumidores validarem
    compatibilidade (colunas/classes esperadas) antes de chamar os demais
    endpoints. Diferente dos outros 3 endpoints, este É totalmente
    implementado nesta task (apenas leitura de metadados já persistidos em
    disco, sem nenhuma inferência)."""

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
        default=None, description="Limiares de camada A/B/C usados no treino do modelo campeão (ver CLASS_TIER_THRESHOLDS)."
    )
    class_weight_registry_available: bool = Field(
        default=False, description="Indica se `model/class_weight_registry.json` está disponível para a futura narrativa via LLM."
    )
    message: str | None = Field(
        default=None,
        description="Mensagem informativa quando nenhum modelo compilado for encontrado (ex.: orientação para rodar scripts/compile_models.py).",
    )
