"""Contrato de resposta do histórico de inferências (Task 010).

``InspecaoLaudoHistoricoEntry`` herda de ``InspecaoLaudoResponse`` (mesmo
padrão de ``InspecaoLaudoCsvItemResponse``): todos os campos da resposta de
análise viram campos de primeiro nível do documento Firestore, lado a lado
com ``usuario_id``/``criado_em``. Não há aninhamento — 1 inferência = 1
documento "achatado" na coleção ``inferencias`` (ver `src/api/services/historico.py`).
"""
from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from .inspecao_response import InspecaoLaudoResponse


class InspecaoLaudoHistoricoEntry(InspecaoLaudoResponse):
    """Item de `GET /historico/{numero_laudo}` — inferência persistida."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "aZ3kP9mQFh2VJmXQ7bLr",
                "numero_laudo": "2025006988",
                "classe_prevista": "10",
                "dsc_classe_prevista": (
                    "O medidor está funcionando de acordo com o Regulamento Técnico Metrológico "
                    "acima referenciado. Os erros percentuais do medidor estão COMPATÍVEIS com "
                    "sua classe de exatidão."
                ),
                "camada": "A",
                "situacao_afericao": "Reprovado",
                "resultado": "Classe 10 (camada A)",
                "resultado_detalhado": "Classe prevista: 10 (camada A). ...",
                "predict_proba": {"10": 0.9123},
                "dsc_predict_proba": {"10": "O medidor está funcionando..."},
                "revisao_llm": None,
                "usuario_id": "fulano@energisa.com.br",
                "criado_em": "2026-09-10T14:32:00Z",
            }
        }
    )

    id: str = Field(..., description="ID do documento Firestore (não gravado como campo — é o Document ID).")
    usuario_id: str = Field(..., description="E-mail de quem gerou a inferência (autenticação obrigatória).")
    criado_em: datetime = Field(..., description="Timestamp de gravação (server-side, Firestore SERVER_TIMESTAMP).")
