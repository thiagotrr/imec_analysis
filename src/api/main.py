"""Aplicação FastAPI IMeC Analysis (definição da app e rotas).

A inicialização do servidor fica em ``src/main.py``:

    python src/main.py          # API (padrão)
    python src/main.py --api    # API
    python src/main.py --ml     # pipeline de ML
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .inspecao_router import router as inspecao_router
from log import get_log

log = get_log()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando API IMeC Analysis")
    try:
        yield
    except Exception:
        log.exception("Erro nao tratado durante o ciclo de vida da API")
        raise
    finally:
        log.info("Encerrando API IMeC Analysis")


app = FastAPI(
    title="IMeC Analysis API",
    description="Inspeção de Medidor de Consumo baseado em laudos analíticos",
    version="0.1.0",
    lifespan=lifespan,
)

# Endpoints da Task 006 (compilação de modelo + contratos + endpoints, ver
# docs/task006_proximos_passos.md): tag "Inspeção de Medidor de Consumo".
app.include_router(inspecao_router)
