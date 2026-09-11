"""Endpoint de consulta ao histórico de inferências (Task 010).

Camada HTTP: valida query params, resolve o client Firestore via
``app.state`` e delega a query ao service. Rota protegida por JWT.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from log import get_log

from ..dependencies.auth import get_current_user
from ..models.auth import AuthenticatedUser
from ..models.inspecao_historico import InspecaoLaudoHistoricoEntry
from ..services import historico as services
from ._common_responses import RESPONSE_401_UNAUTHORIZED

log = get_log()

TAG = "Histórico de Inferências"

router = APIRouter()

_RESPONSE_503_FIRESTORE = {"description": "Firestore indisponível."}

_DATA_INICIO_QUERY = Query(default=None, description="Filtra inferências gravadas a partir desta data (inclusive).")
_DATA_FIM_QUERY = Query(default=None, description="Filtra inferências gravadas até esta data (inclusive).")


@router.get(
    "/historico/{numero_laudo}",
    tags=[TAG],
    summary="Consulta o histórico de inferências de um laudo",
    description=(
        "Retorna as inferências já persistidas para `numero_laudo`, mais recente primeiro. "
        "`data_inicio`/`data_fim` (opcionais) filtram por período de gravação (`criado_em`). "
        "Retorna lista vazia (200) quando o laudo não tem inferências persistidas — não é erro."
    ),
    response_model=list[InspecaoLaudoHistoricoEntry],
    responses={401: RESPONSE_401_UNAUTHORIZED, 503: _RESPONSE_503_FIRESTORE},
)
def consultar_historico(
    numero_laudo: str,
    request: Request,
    data_inicio: datetime | None = _DATA_INICIO_QUERY,
    data_fim: datetime | None = _DATA_FIM_QUERY,
    usuario: AuthenticatedUser = Depends(get_current_user),
) -> list[InspecaoLaudoHistoricoEntry]:
    log.info("Consulta de histórico (NUMLAUDO=%s, usuario=%s)", numero_laudo, usuario.email)
    firestore_runtime = getattr(request.app.state, "firestore", None)
    if firestore_runtime is None:
        raise HTTPException(status_code=503, detail="Firestore indisponível no momento.")

    try:
        return services.listar_historico(
            firestore_runtime.client, numero_laudo, data_inicio=data_inicio, data_fim=data_fim
        )
    except Exception as exc:
        log.exception("Falha ao consultar histórico em /historico/%s", numero_laudo)
        raise HTTPException(status_code=503, detail=f"Falha ao consultar histórico: {exc}") from exc
