"""Persistência e consulta de histórico de inferências, coleção ``inferencias`` (Task 010).

Funções puras que recebem o client Firestore como parâmetro (mesmo padrão de
``services/inspecao.py`` recebendo ``runtime``), não o buscam sozinhas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from google.cloud.firestore_v1 import Query

from ..models.inspecao_historico import InspecaoLaudoHistoricoEntry
from ..models.inspecao_response import InspecaoLaudoResponse

COLLECTION = "inferencias"


def persistir_inferencia(
    client: Any,
    response: InspecaoLaudoResponse,
    *,
    usuario_id: str,
) -> InspecaoLaudoHistoricoEntry:
    """Grava 1 documento "achatado" (campos de ``response`` + metadados) em
    ``inferencias/{auto_id}``. Doc ID auto-gerado — o mesmo laudo pode ser
    reanalisado várias vezes; o histórico nunca sobrescreve entradas
    anteriores."""
    payload = response.model_dump(mode="json")
    payload["usuario_id"] = usuario_id
    payload["criado_em"] = datetime.now().astimezone()

    _, doc_ref = client.collection(COLLECTION).add(payload)
    return InspecaoLaudoHistoricoEntry(id=doc_ref.id, **payload)


def listar_historico(
    client: Any,
    numero_laudo: str,
    *,
    data_inicio: datetime | None = None,
    data_fim: datetime | None = None,
    limit: int = 50,
) -> list[InspecaoLaudoHistoricoEntry]:
    """Consulta por ``numero_laudo`` (obrigatório), com filtro opcional de
    período sobre ``criado_em``, mais recente primeiro."""
    query = (
        client.collection(COLLECTION)
        .where("numero_laudo", "==", numero_laudo)
        .order_by("criado_em", direction=Query.DESCENDING)
    )
    if data_inicio is not None:
        query = query.where("criado_em", ">=", data_inicio)
    if data_fim is not None:
        query = query.where("criado_em", "<=", data_fim)
    query = query.limit(limit)

    entries: list[InspecaoLaudoHistoricoEntry] = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        data["id"] = doc.id
        entries.append(InspecaoLaudoHistoricoEntry(**data))
    return entries
