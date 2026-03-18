import sys
import uvicorn
from log import get_log

log = get_log()

def main():
    log.info("Iniciando API IMeC Analysis", extra={"event": "api_start"})
    try:
        uvicorn.run(
            "api.main:app", 
            host="0.0.0.0", 
            port=8000,
            reload=True
            )
    except Exception as e:
        log.exception("Erro não tratado durante a execução da API", extra={"event": "api_error"})
        sys.exit(1)
    
if __name__ == "__main__":
    main()