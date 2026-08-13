# Imagem da API IMeC Analysis: serve inferência a partir dos PKLs já compilados.
# Não executa pipeline de treino/preparação.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY model ./model

ENV PYTHONPATH=/app/src
ENV IMEC_API_RELOAD=false
# Default seguro: LLM desligada até existir secret no Cloud Run.
# Nunca coloque OPENAI_API_KEY / GEMINI_API_KEY neste arquivo.
# Local: .env (gitignored). Cloud Run: Secret Manager (--set-secrets).
ENV LLM_ENABLED=false
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn --app-dir src api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
