from contextlib import asynccontextmanager
from fastapi import FastAPI
from .inspecao_router import router as inspecao_router
from .request_model import InspecaoMedidorRequest
from .response_model import InspecaoMedidorResponse
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
# docs/task006_proximos_passos.md): tag própria ("Inspeção de Medidor de
# Consumo"), separada da tag do endpoint legado abaixo (`/analise_inspecao`).
app.include_router(inspecao_router)

@app.post("/analise_inspecao", 
          tags=["Análise de Inspeção de Medidor de Consumo"],
          summary="Realiza a análise de inspeção de um medidor de consumo",
          description="Recebe os dados de um laudo analítico do INMETRO e retorna uma avaliação da conformidade do medidor de consumo.",
          response_model=InspecaoMedidorResponse)
def analisar_inspecao(inspecao_request: InspecaoMedidorRequest) -> InspecaoMedidorResponse:
    log.info(f"Recebida solicitação de análise para medidor ID {inspecao_request.id_medidor} na data {inspecao_request.data_inspecao}")
    
    log.info(f"Análise concluída com sucesso para medidor ID {inspecao_request.id_medidor}")
    return InspecaoMedidorResponse(
        id_medidor=inspecao_request.id_medidor,
        data_inspecao=inspecao_request.data_inspecao,
        resultado="Teste unitário TRR aprovado",
        resultado_detalhado="Detalhamento do resultado da análise do laudo"
    )