"""Endpoints de Inspeção de Medidor de Consumo (Task 006, §3).

Cobre apenas a camada HTTP: validação de payload via os contratos Pydantic
de `inspecao_request_model.py` (automática pelo FastAPI) e chamada a um
service placeholder (`inspecao_services.py`) que levanta `NotImplementedError`
— convertido aqui em `HTTPException(500, ...)`, documentado no `description`
de cada endpoint. Nenhuma lógica de inferência real é implementada (ver
docs/task006_proximos_passos.md, §4 "Fora do escopo desta task").

A única exceção é `GET /inspecao/modelos`, que é totalmente funcional: apenas
lê metadados já persistidos em `model/compiled/*.json` (ver
`inspecao_services.obter_info_modelos`), sem inferência.
"""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import ValidationError

from log import get_log

from . import inspecao_services as services
from .inspecao_request_model import LaudoCompletoRequest, LaudoSinteticoRequest, validate_laudo_completo_row
from .inspecao_response_model import InspecaoLaudoCsvItemResponse, InspecaoLaudoResponse, ModeloInfoResponse

log = get_log()

TAG = "Inspeção de Medidor de Consumo"

router = APIRouter()

_RESPONSE_422_MODEL_INVALID = {
    "description": "Payload inválido: um ou mais campos não correspondem ao contrato esperado (tipo, obrigatoriedade ou campo desconhecido).",
}
_RESPONSE_500_SERVICE_NOT_IMPLEMENTED = {
    "description": (
        "Service de inferência ainda não implementado nesta task — ver "
        "docs/task006_proximos_passos.md, §4 ('Fora do escopo desta task')."
    ),
}


def _service_not_implemented_to_http(exc: NotImplementedError, endpoint: str) -> HTTPException:
    log.exception("Service de %s ainda não implementado", endpoint)
    return HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/inspecao/laudo_completo",
    tags=[TAG],
    summary="Analisa um laudo completo (todas as colunas do dataset)",
    description=(
        "Recebe TODAS as colunas do laudo de aferição (mesmo layout de "
        "`resultado_laudo_afericao.xlsx`, exceto o target `CODRSTAFER`) e retornaria a "
        "classificação do medidor. Uso: consumidor que já possui o laudo completo e não "
        "quer se preocupar em saber quais colunas o modelo de fato usa (ver "
        "docs/task006_proximos_passos.md, §2.1). "
        "\n\n**TODO**: a inferência real (carregar `model/compiled/champion.pkl`, "
        "pré-processar via `preprocessing_pipeline.pkl` e decodificar via `target_encoder.pkl`) "
        "ainda não está implementada — ver §4 do plano. Esta chamada validará o payload "
        "(422 em caso de schema inválido) e retornará 500 (service não implementado) em caso "
        "de payload válido."
    ),
    response_model=InspecaoLaudoResponse,
    responses={422: _RESPONSE_422_MODEL_INVALID, 500: _RESPONSE_500_SERVICE_NOT_IMPLEMENTED},
)
def analisar_laudo_completo(laudo: LaudoCompletoRequest) -> InspecaoLaudoResponse:
    log.info("Recebida solicitação de análise de laudo completo (NUMLAUDO=%s)", getattr(laudo, "NUMLAUDO", None))
    try:
        return services.analisar_laudo_completo(laudo)
    except NotImplementedError as exc:
        raise _service_not_implemented_to_http(exc, "análise de laudo completo") from exc


@router.post(
    "/inspecao/laudo_sintetico",
    tags=[TAG],
    summary="Analisa um laudo sintético (somente as features usadas pelo modelo)",
    description=(
        "Recebe apenas as features retidas pelo último treino definitivo "
        "(`retained_feature_columns` em `model/preparation_metadata.json` — contrato "
        "fixo `LaudoSinteticoRequest`). Uso: integração magra com o mínimo necessário "
        "para o `preprocessing_pipeline.pkl` ser executável. "
        "\n\n**TODO**: mesma ressalva do endpoint `/inspecao/laudo_completo` — inferência "
        "real fora do escopo desta task (ver §4 do plano)."
    ),
    response_model=InspecaoLaudoResponse,
    responses={422: _RESPONSE_422_MODEL_INVALID, 500: _RESPONSE_500_SERVICE_NOT_IMPLEMENTED},
)
def analisar_laudo_sintetico(laudo: LaudoSinteticoRequest) -> InspecaoLaudoResponse:
    log.info("Recebida solicitação de análise de laudo sintético (NUMLAUDO=%s)", getattr(laudo, "NUMLAUDO", None))
    try:
        return services.analisar_laudo_sintetico(laudo)
    except NotImplementedError as exc:
        raise _service_not_implemented_to_http(exc, "análise de laudo sintético") from exc


@router.post(
    "/inspecao/csv",
    tags=[TAG],
    summary="Analisa em lote um CSV com um ou mais laudos. Encoding UTF-8.",
    description=(
        "Recebe um arquivo CSV (multipart/form-data) no mesmo layout de "
        "`resultado_laudo_afericao.xlsx` (mesmas colunas do contrato `LaudoCompletoRequest`, "
        "uma ou mais linhas). Cada linha é validada individualmente reaproveitando o schema "
        "de `LaudoCompletoRequest` (ver docs/task006_proximos_passos.md, §2.3); se qualquer "
        "linha for inválida, a resposta é 422 com o detalhamento por linha/coluna, sem "
        "processar nenhuma linha do lote. "
        "\n\n**TODO**: mesma ressalva dos demais endpoints de análise — inferência real fora "
        "do escopo desta task (ver §4 do plano)."
    ),
    response_model=list[InspecaoLaudoCsvItemResponse],
    responses={422: _RESPONSE_422_MODEL_INVALID, 500: _RESPONSE_500_SERVICE_NOT_IMPLEMENTED},
)
async def analisar_csv_upload(arquivo: UploadFile) -> list[InspecaoLaudoCsvItemResponse]:
    raw_bytes = await arquivo.read()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        log.exception("Falha ao decodificar o CSV enviado em /inspecao/csv")
        raise HTTPException(status_code=422, detail=f"Não foi possível decodificar o arquivo como UTF-8: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=422, detail="CSV vazio ou sem linhas de dados (apenas cabeçalho/nenhum conteúdo).")

    validated_laudos: list[LaudoCompletoRequest] = []
    row_errors: list[dict[str, object]] = []
    for line_number, row in enumerate(rows, start=1):
        cleaned_row = {key: (value if value != "" else None) for key, value in row.items()}
        try:
            validated_laudos.append(validate_laudo_completo_row(cleaned_row))
        except ValidationError as exc:
            row_errors.append({"numero_linha": line_number, "erros": exc.errors()})

    if row_errors:
        log.info("Upload CSV em /inspecao/csv rejeitado: %d linha(s) inválida(s)", len(row_errors))
        raise HTTPException(status_code=422, detail={"linhas_invalidas": row_errors})

    log.info("Recebida solicitação de análise em lote via CSV (%d linha(s))", len(validated_laudos))
    try:
        return services.analisar_csv_upload(validated_laudos)
    except NotImplementedError as exc:
        raise _service_not_implemented_to_http(exc, "análise em lote via upload CSV") from exc


@router.get(
    "/inspecao/modelos",
    tags=[TAG],
    summary="Lista metadados do(s) modelo(s) compilado(s) disponíveis",
    description=(
        "Retorna os metadados do modelo campeão e das demais combinações algoritmo+resampling "
        "compiladas em `model/compiled/` (ver `scripts/compile_models.py`): algoritmo, "
        "resampling, métricas de validação, colunas de entrada esperadas e classes suportadas. "
        "Útil para consumidores validarem compatibilidade antes de chamar os demais endpoints. "
        "Diferente dos demais endpoints desta tag, este É totalmente funcional (apenas leitura "
        "de metadados já persistidos em disco — nenhuma inferência é executada). Se nenhum "
        "modelo tiver sido compilado ainda, retorna 200 com `champion=null` e uma mensagem "
        "explicativa (não é tratado como erro)."
    ),
    response_model=ModeloInfoResponse,
    responses={500: {"description": "Falha inesperada ao ler os metadados em model/compiled/."}},
)
def listar_modelos() -> ModeloInfoResponse:
    try:
        return services.obter_info_modelos()
    except Exception as exc:  # pragma: no cover - defensivo (I/O inesperado)
        log.exception("Erro ao carregar metadados dos modelos compilados")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar metadados dos modelos compilados: {exc}") from exc
