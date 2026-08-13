"""Aplicação FastAPI IMeC Analysis (definição da app e rotas).

A inicialização do servidor fica em ``src/main.py``:

    python src/main.py          # API (padrão)
    python src/main.py --api    # API
    python src/main.py --ml     # pipeline de ML
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request

from llm.reviewer import build_default_reviewer
from log import get_log

from .model_runtime import ModelRuntimeError, load_model_runtime
from .routers.inspecao import router as inspecao_router
from .services.inspecao import load_llm_dependencies

log = get_log()

# Carrega `.env` na raiz do repositório (OPENAI_API_KEY / GEMINI_API_KEY / LLM_*).
load_dotenv()


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

    llm_deps = load_llm_dependencies()
    app.state.llm_settings = llm_deps["llm_settings"]
    app.state.glossary = llm_deps["glossary"]
    app.state.class_metrics_lookup = llm_deps["class_metrics_lookup"]
    app.state.llm_reviewer = build_default_reviewer(app.state.llm_settings)
    log.info(
        "LLM pós-processamento: enabled=%s provider=%s configured=%s glossary_status=%s",
        app.state.llm_settings.enabled,
        app.state.llm_settings.provider,
        app.state.llm_settings.is_configured(),
        getattr(app.state.glossary, "status", "unknown"),
    )

    try:
        yield
    except Exception:
        log.exception("Erro nao tratado durante o ciclo de vida da API")
        raise
    finally:
        app.state.runtime = None
        app.state.llm_reviewer = None
        log.info("Encerrando API IMeC Analysis")


app = FastAPI(
    title="IMeC Analysis API",
    description="Inspeção de Medidor de Consumo baseado em laudos analíticos",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(inspecao_router)


@app.get(
    "/health",
    tags=["Infra"],
    summary="Saúde da API e carga do modelo compilado",
    description=(
        "Indica se o processo está no ar e se o runtime de inferência "
        "(champion.pkl + preprocessing_pipeline.pkl) foi carregado no startup. "
        "Sempre responde 200: `status=degraded` quando os artefatos PKL não "
        "estiverem disponíveis — útil como health check de PaaS sem derrubar "
        "o container."
    ),
)
def health(request: Request) -> dict[str, object]:
    runtime = getattr(request.app.state, "runtime", None)
    metadata = getattr(runtime, "champion_metadata", None) or {}
    return {
        "status": "ok" if runtime is not None else "degraded",
        "runtime_loaded": runtime is not None,
        "champion_algorithm": metadata.get("algorithm"),
        "champion_resampling": metadata.get("resampling"),
    }
