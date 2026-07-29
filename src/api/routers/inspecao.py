"""Endpoints de Inspeção de Medidor de Consumo (Task 006/007/008).

Camada HTTP: validação Pydantic + chamada aos services de inferência.
Erros de runtime/inferência (`ModelRuntimeError`) → HTTP 500.

Query ``revisao_llm`` (opcional) força/desliga a revisão LLM nos endpoints
unitários. CSV em lote nunca chama LLM.
"""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from log import get_log

from api.services import inspecao as services
from api.models.inspecao_request import LaudoCompletoRequest, LaudoSinteticoRequest, validate_laudo_completo_row
from api.models.inspecao_response import InspecaoLaudoCsvItemResponse, InspecaoLaudoResponse, ModeloInfoResponse
from api.model_runtime import ModelRuntimeError

log = get_log()

TAG = "Inspeção de Medidor de Consumo"

router = APIRouter()

_RESPONSE_422_MODEL_INVALID = {
    "description": "Payload inválido: um ou mais campos não correspondem ao contrato esperado (tipo, obrigatoriedade ou campo desconhecido).",
}
_RESPONSE_500_INFERENCE = {
    "description": (
        "Falha de inferência: artefatos ausentes/não carregados no startup, "
        "erro ao pré-processar ou ao executar o modelo campeão."
    ),
}

_REVISAO_LLM_QUERY = Query(
    default=None,
    description=(
        "Controla a revisão LLM pós-inferência. "
        "`true` força a chamada (mesmo em camada A); "
        "`false` desliga; "
        "omitido aplica o gate padrão (não chama para camada A / classes majoritárias)."
    ),
)


def _runtime_from_request(request: Request):
    return getattr(request.app.state, "runtime", None)


def _llm_kwargs_from_request(request: Request, revisao_llm: bool | None) -> dict:
    state = request.app.state
    return {
        "revisao_llm": revisao_llm,
        "llm_reviewer": getattr(state, "llm_reviewer", None),
        "llm_settings": getattr(state, "llm_settings", None),
        "glossary": getattr(state, "glossary", None),
        "class_metrics_lookup": getattr(state, "class_metrics_lookup", None),
    }


def _inference_error_to_http(exc: ModelRuntimeError, endpoint: str) -> HTTPException:
    log.exception("Falha de inferência em %s", endpoint)
    return HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/inspecao/laudo_completo",
    tags=[TAG],
    summary="Analisa um laudo completo (todas as colunas do dataset)",
    description=(
        "Recebe TODAS as colunas do laudo de aferição (mesmo layout de "
        "`resultado_laudo_afericao.xlsx`, exceto o target `CODRSTAFER`), filtra as "
        "features retidas, executa `preprocessing_pipeline` + `champion` e retorna "
        "classe prevista, camada (A–D), `predict_proba`, texto template e, quando o "
        "gate LLM permitir, `revisao_llm`."
    ),
    response_model=InspecaoLaudoResponse,
    responses={422: _RESPONSE_422_MODEL_INVALID, 500: _RESPONSE_500_INFERENCE},
)
def analisar_laudo_completo(
    laudo: LaudoCompletoRequest,
    request: Request,
    revisao_llm: bool | None = _REVISAO_LLM_QUERY,
) -> InspecaoLaudoResponse:
    log.info("Recebida solicitação de análise de laudo completo (NUMLAUDO=%s)", getattr(laudo, "NUMLAUDO", None))
    try:
        return services.analisar_laudo_completo(
            laudo,
            _runtime_from_request(request),
            **_llm_kwargs_from_request(request, revisao_llm),
        )
    except ModelRuntimeError as exc:
        raise _inference_error_to_http(exc, "análise de laudo completo") from exc


@router.post(
    "/inspecao/laudo_sintetico",
    tags=[TAG],
    summary="Analisa um laudo sintético (somente as features usadas pelo modelo)",
    description=(
        "Recebe apenas as features retidas pelo último treino definitivo "
        "(`retained_feature_columns` — contrato `LaudoSinteticoRequest`) e executa "
        "a mesma inferência do endpoint de laudo completo, com revisão LLM opcional."
    ),
    response_model=InspecaoLaudoResponse,
    responses={422: _RESPONSE_422_MODEL_INVALID, 500: _RESPONSE_500_INFERENCE},
)
def analisar_laudo_sintetico(
    laudo: LaudoSinteticoRequest,
    request: Request,
    revisao_llm: bool | None = _REVISAO_LLM_QUERY,
) -> InspecaoLaudoResponse:
    log.info("Recebida solicitação de análise de laudo sintético (NUMLAUDO=%s)", getattr(laudo, "NUMLAUDO", None))
    try:
        return services.analisar_laudo_sintetico(
            laudo,
            _runtime_from_request(request),
            **_llm_kwargs_from_request(request, revisao_llm),
        )
    except ModelRuntimeError as exc:
        raise _inference_error_to_http(exc, "análise de laudo sintético") from exc


@router.post(
    "/inspecao/csv",
    tags=[TAG],
    summary="Analisa em lote um CSV com um ou mais laudos. Encoding UTF-8.",
    description=(
        "Recebe um arquivo CSV (multipart/form-data) no layout de "
        "`LaudoCompletoRequest`. Cada linha é validada individualmente; se qualquer "
        "linha for inválida, a resposta é 422 sem processar o lote. Caso contrário, "
        "executa inferência por linha e devolve `numero_linha` em cada item. "
        "Não dispara revisão LLM (custo/latência); use os endpoints unitários."
    ),
    response_model=list[InspecaoLaudoCsvItemResponse],
    responses={422: _RESPONSE_422_MODEL_INVALID, 500: _RESPONSE_500_INFERENCE},
)
async def analisar_csv_upload(arquivo: UploadFile, request: Request) -> list[InspecaoLaudoCsvItemResponse]:
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
        return services.analisar_csv_upload(validated_laudos, _runtime_from_request(request))
    except ModelRuntimeError as exc:
        raise _inference_error_to_http(exc, "análise em lote via upload CSV") from exc


@router.get(
    "/inspecao/modelos",
    tags=[TAG],
    summary="Lista metadados do(s) modelo(s) compilado(s) disponíveis",
    description=(
        "Retorna os metadados do modelo campeão e das demais combinações algoritmo+resampling "
        "compiladas em `model/compiled/`. Se nenhum modelo tiver sido compilado ainda, retorna "
        "200 com `champion=null` e uma mensagem explicativa (não é tratado como erro)."
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
