"""Ponto de entrada unificado do projeto IMeC Analysis.

Uso:
    python src/main.py           # sobe a API (padrão)
    python src/main.py --api     # sobe a API
    python src/main.py --ml      # pipeline de preparação + classificação
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from log import get_log
from machine_learning import print_preparation_summary, run_classification_workflow, run_preparation_workflow

log = get_log()

DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8000
DEFAULT_HTTPS_PORT = 8443


def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _resolve_tls_path(path_value: str, project_root: Path) -> tuple[str | None, list[Path]]:
    raw = Path(path_value)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [Path.cwd() / raw, project_root / raw]
        # Conveniência: se vier só o nome do arquivo, tenta também em ./certs.
        if raw.parent == Path("."):
            candidates.append(project_root / "certs" / raw.name)

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False)).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve()), unique_candidates

    return None, unique_candidates


def _resolve_api_run_config(args: argparse.Namespace) -> dict[str, object]:
    env = os.environ
    project_root = Path(__file__).resolve().parent.parent

    https_enabled = (
        args.https
        if args.https is not None
        else _parse_bool_env(env.get("IMEC_API_HTTPS"), default=False)
    )

    host = args.host or env.get("IMEC_API_HOST") or DEFAULT_API_HOST

    if args.port is not None:
        port = args.port
    elif env.get("IMEC_API_PORT"):
        port = int(env["IMEC_API_PORT"])
    elif env.get("PORT"):
        # Cloud Run, Render, Koyeb e similares injetam PORT.
        port = int(env["PORT"])
    else:
        port = DEFAULT_HTTPS_PORT if https_enabled else DEFAULT_HTTP_PORT

    if not (1 <= port <= 65535):
        raise ValueError(f"Porta inválida: {port}. Use um valor entre 1 e 65535.")

    # Em PaaS a variável PORT está presente: reload de desenvolvimento não deve ligar.
    default_reload = not bool(env.get("PORT"))
    reload_enabled = (
        args.reload
        if args.reload is not None
        else _parse_bool_env(env.get("IMEC_API_RELOAD"), default=default_reload)
    )

    ssl_certfile = args.ssl_certfile or env.get("IMEC_API_SSL_CERTFILE")
    ssl_keyfile = args.ssl_keyfile or env.get("IMEC_API_SSL_KEYFILE")
    ssl_keyfile_password = args.ssl_keyfile_password or env.get("IMEC_API_SSL_KEYFILE_PASSWORD")

    if https_enabled:
        if not ssl_certfile or not ssl_keyfile:
            raise ValueError(
                "HTTPS habilitado, mas certificado/chave não informados. "
                "Use --ssl-certfile e --ssl-keyfile (ou variáveis IMEC_API_SSL_CERTFILE/IMEC_API_SSL_KEYFILE)."
            )

        resolved_certfile, cert_candidates = _resolve_tls_path(ssl_certfile, project_root)
        if resolved_certfile is None:
            attempts = ", ".join(str(path.resolve(strict=False)) for path in cert_candidates)
            raise ValueError(
                "Arquivo de certificado TLS não encontrado. "
                f"Valor recebido: '{ssl_certfile}'. Caminhos tentados: {attempts}."
            )

        resolved_keyfile, key_candidates = _resolve_tls_path(ssl_keyfile, project_root)
        if resolved_keyfile is None:
            attempts = ", ".join(str(path.resolve(strict=False)) for path in key_candidates)
            raise ValueError(
                "Arquivo de chave TLS não encontrado. "
                f"Valor recebido: '{ssl_keyfile}'. Caminhos tentados: {attempts}."
            )

        ssl_certfile = resolved_certfile
        ssl_keyfile = resolved_keyfile

        if port == DEFAULT_HTTP_PORT:
            raise ValueError(
                "HTTPS não deve usar a porta padrão HTTP (8000). "
                "Use --port 443, 8443 ou outra porta apropriada para HTTPS."
            )

    return {
        "host": host,
        "port": port,
        "reload": reload_enabled,
        "https_enabled": https_enabled,
        "ssl_certfile": ssl_certfile,
        "ssl_keyfile": ssl_keyfile,
        "ssl_keyfile_password": ssl_keyfile_password,
    }


def run_api(args: argparse.Namespace) -> None:
    log.info("Iniciando API IMeC Analysis", extra={"event": "api_start"})
    try:
        src_dir = Path(__file__).resolve().parent
        config = _resolve_api_run_config(args)
        scheme = "https" if bool(config["https_enabled"]) else "http"
        log.info(
            "Servidor API configurado: %s://%s:%s (reload=%s)",
            scheme,
            config["host"],
            config["port"],
            config["reload"],
        )

        uvicorn_kwargs: dict[str, object] = {
            "host": config["host"],
            "port": config["port"],
            "reload": config["reload"],
            "app_dir": str(src_dir),
        }
        if bool(config["https_enabled"]):
            uvicorn_kwargs["ssl_certfile"] = config["ssl_certfile"]
            uvicorn_kwargs["ssl_keyfile"] = config["ssl_keyfile"]
            if config["ssl_keyfile_password"]:
                uvicorn_kwargs["ssl_keyfile_password"] = config["ssl_keyfile_password"]

        uvicorn.run(
            "api.main:app",
            **uvicorn_kwargs,
        )
    except Exception:
        log.exception("Erro não tratado durante a execução da API", extra={"event": "api_error"})
        sys.exit(1)


def run_ml_pipeline(output_dir: str | Path | None = None) -> None:
    log.info("Iniciando pipeline de machine learning", extra={"event": "ml_pipeline_start"})
    try:
        workflow_result = run_preparation_workflow(output_dir=output_dir, open_browser=True)
        print_preparation_summary(workflow_result.preparation_result)

        if workflow_result.dashboard_path is not None:
            print(f"Dashboard HTML: {workflow_result.dashboard_path}")
            log.info(
                "Dashboard HTML gerado e solicitado ao browser",
                extra={"event": "exploration_dashboard", "path": str(workflow_result.dashboard_path)},
            )

        classification_result = run_classification_workflow(output_dir=output_dir)
        if classification_result.artifacts is not None:
            print(f"Resumo consolidado: {classification_result.artifacts.summary_path}")
    except Exception:
        log.exception(
            "Erro não tratado durante a execução do pipeline de machine learning",
            extra={"event": "ml_pipeline_error"},
        )
        sys.exit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IMeC Analysis — entrada unificada (API ou pipeline de ML).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--api",
        action="store_true",
        help="Inicia a API FastAPI (também é o comportamento padrão sem argumentos).",
    )
    mode.add_argument(
        "--ml",
        action="store_true",
        help="Executa o pipeline de preparação + exploração + classificação.",
    )

    parser.add_argument(
        "--host",
        help=(
            "Host de bind do uvicorn. Padrão: 0.0.0.0 "
            "(ou IMEC_API_HOST)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "Porta do uvicorn. Padrão: 8000 (HTTP) ou 8443 (HTTPS). "
            "Também aceita IMEC_API_PORT ou PORT (PaaS)."
        ),
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Ativa/desativa reload automático (padrão: ligado localmente; "
            "desligado quando PORT está definido, típico de PaaS). "
            "Também IMEC_API_RELOAD."
        ),
    )
    parser.add_argument(
        "--https",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ativa/desativa HTTPS (também IMEC_API_HTTPS).",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="Caminho do certificado TLS PEM (também IMEC_API_SSL_CERTFILE).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Caminho da chave privada TLS PEM (também IMEC_API_SSL_KEYFILE).",
    )
    parser.add_argument(
        "--ssl-keyfile-password",
        help="Senha da chave privada TLS (também IMEC_API_SSL_KEYFILE_PASSWORD).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.ml:
        run_ml_pipeline()
        return

    # Sem argumentos ou com --api → API
    run_api(args)


if __name__ == "__main__":
    main()
