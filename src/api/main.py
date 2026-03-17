from contextlib import asynccontextmanager
from fastapi import FastAPI
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
    description="Inspeção de Medidor de Consumo baseado em laudos analíticos do INMETRO",
    version="0.1.0",
    lifespan=lifespan,
)