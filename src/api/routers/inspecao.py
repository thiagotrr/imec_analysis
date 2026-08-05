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
    "description": (
        "Dados enviados inválidos: um ou mais campos estão fora do formato esperado "
        "(tipo incorreto, campo obrigatório ausente ou campo não reconhecido)."
    ),
}
_RESPONSE_500_INFERENCE = {
    "description": (
        "Falha ao analisar o laudo: arquivos do modelo ausentes/não carregados na "
        "inicialização, erro no preparo dos dados ou erro na execução do modelo "
        "principal (champion)."
    ),
}

_REVISAO_LLM_QUERY = Query(
    default=None,
    description=(
        "Define se a revisão por IA (revisao_llm) será aplicada após a predição. "
        "`true` força a revisão (inclusive em camada A); "
        "`false` desliga a revisão; "
        "se não informar, usa a regra automática (gate padrão), que não revisa camada "
        "A nem classes mais frequentes."
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
        "Recebe todas as colunas do laudo (mesmo layout de "
        "`resultado_laudo_afericao.xlsx`), exceto o campo resultado da aferição "
        "(`CODRSTAFER`). Em seguida, seleciona os campos usados pelo modelo "
        "(features retidas), prepara os dados (`preprocessing_pipeline`) e executa o "
        "modelo principal (`champion`). Retorna o resultado previsto, a camada de "
        "prioridade (A a D), a probabilidade de acerto (`predict_proba`), o texto "
        "padrão da análise e, quando permitido, a revisão por IA (`revisao_llm`)."
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
        "Recebe somente os campos que o modelo realmente usa (features retidas "
        "`retained_feature_columns`, no contrato `LaudoSinteticoRequest`) e executa "
        "a mesma análise do laudo completo, com revisão por IA (`revisao_llm`) "
        "opcional."
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
        "Recebe um arquivo CSV (multipart/form-data) no formato de "
        "`LaudoCompletoRequest`. Cada linha é validada separadamente. Se houver "
        "qualquer linha inválida, o lote é interrompido e a API retorna 422 com o "
        "detalhe das linhas com erro. Se tudo estiver válido, a análise é feita "
        "linha a linha e cada item da resposta informa o número da linha "
        "(`numero_linha`). Para reduzir custo e tempo de resposta, este endpoint não "
        "executa revisão por IA (LLM)."
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
        "Retorna os metadados dos modelos já compilados: modelo principal "
        "(`champion`) e outras combinações de algoritmo com balanceamento "
        "(`resampling`) salvas em `model/compiled/`. Se ainda não existir modelo "
        "compilado, retorna 200 com `champion=null` e uma mensagem explicativa "
        "(não é erro)."
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
