"""Pacote da API FastAPI de Inspeção de Medidor de Consumo (IMeC Analysis).

Layout:

- ``main`` — app FastAPI + lifespan
- ``routers/`` — endpoints HTTP
- ``models/`` — contratos Pydantic (request/response)
- ``services/`` — orquestração de inferência / LLM
- ``inference_pipeline`` / ``model_runtime`` — runtime de ML
- ``schema_generation`` — utilitário de schemas (offline)
"""
