import sys
from pathlib import Path

import uvicorn
from log import get_log
from machine_learning import print_preparation_summary, run_preparation_workflow

log = get_log()

def run_api() -> None:
    log.info("Iniciando API IMeC Analysis", extra={"event": "api_start"})
    try:
        uvicorn.run(
            "api.main:app", 
            host="0.0.0.0", 
            port=8000,
            reload=True
            )
    except Exception:
        log.exception("Erro não tratado durante a execução da API", extra={"event": "api_error"})
        sys.exit(1)


def run_preparation_and_show_exploration(output_dir: str | Path | None = None) -> None:
    log.info("Iniciando fluxo de preparação e exploração gráfica", extra={"event": "exploration_start"})
    try:
        workflow_result = run_preparation_workflow(output_dir=output_dir, open_browser=True)
        print_preparation_summary(workflow_result.preparation_result)

        if workflow_result.dashboard_path is not None:
            print(f"Dashboard HTML: {workflow_result.dashboard_path}")
            log.info(
                "Dashboard HTML gerado e solicitado ao browser",
                extra={"event": "exploration_dashboard", "path": str(workflow_result.dashboard_path)},
            )
    except Exception:
        log.exception("Erro não tratado durante a execução do fluxo de exploração", extra={"event": "exploration_error"})
        sys.exit(1)


def main() -> None:
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "exploration"

    if command == "api":
        run_api()
        return

    run_preparation_and_show_exploration()


if __name__ == "__main__":
    main()