"""Carga e cache do client Firestore (Task 010).

Arcabouço isolado e reutilizável: não conhece "inspeção", "auth" nem
qualquer outra vertical do produto — expõe apenas um client Firestore
carregado uma vez no ``lifespan`` da API e guardado em ``app.state.firestore``,
mais um helper genérico de acesso a coleção. Quem decide nomes de coleção e
schema dos documentos são os módulos consumidores (``src/auth``,
``src/api/services/historico.py``).

Credenciais: resolvidas via Application Default Credentials (ADC). No Cloud
Run, a service account do serviço fornece ADC automaticamente através do
metadata server — nenhuma variável ``GOOGLE_APPLICATION_CREDENTIALS`` nem
chave JSON é necessária no código ou no container.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.cloud import firestore

from .settings import FirestoreSettings, load_firestore_settings


class FirestoreRuntimeError(RuntimeError):
    """Falha ao construir o client Firestore ou runtime indisponível."""


@dataclass(frozen=True)
class FirestoreRuntime:
    """Client Firestore pronto para uso, carregado no startup."""

    client: Any
    settings: FirestoreSettings


def load_firestore_runtime() -> FirestoreRuntime:
    """Constrói o client Firestore uma vez (fail-soft: erros de configuração
    viram ``FirestoreRuntimeError``; a construção do ``Client`` é lazy — falhas
    reais de conectividade só aparecem na primeira chamada de leitura/escrita)."""
    settings = load_firestore_settings()
    if not settings.enabled:
        raise FirestoreRuntimeError("Firestore desabilitado (FIRESTORE_ENABLED=false).")
    try:
        client = firestore.Client(project=settings.project_id, database=settings.database_id)
    except Exception as exc:  # biblioteca do Google levanta tipos variados (ValueError, DefaultCredentialsError)
        raise FirestoreRuntimeError(f"Falha ao inicializar o client Firestore: {exc}") from exc
    return FirestoreRuntime(client=client, settings=settings)


def get_collection(runtime: FirestoreRuntime, name: str) -> Any:
    """Único helper genérico exposto por este módulo."""
    return runtime.client.collection(name)
