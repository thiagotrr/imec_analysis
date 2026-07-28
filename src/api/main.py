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
from .model_runtime import ModelRuntimeError, load_model_runtime
from log import get_log

log = get_log()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando API IMeC Analysis")
    # Decisão Task 007: sem pkl compilado a API sobe; rotas de análise devolvem 500.
    # GET /inspecao/modelos continua funcional (só lê metadados JSON).
    try:
        app.state.runtime = load_model_runtime()
        log.info(
            "Runtime de inferência carregado (champion + preprocessing_pipeline + class_weight_registry)"
        )
    except ModelRuntimeError as exc:
        app.state.runtime = None
        log.error("Runtime de inferência indisponível no startup: %s", exc)
    try:
        yield
    except Exception:
        log.exception("Erro nao tratado durante o ciclo de vida da API")
        raise
    finally:
        app.state.runtime = None
        log.info("Encerrando API IMeC Analysis")


app = FastAPI(
    title="IMeC Analysis API",
    description="Inspeção de Medidor de Consumo baseado em laudos analíticos",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(inspecao_router)
