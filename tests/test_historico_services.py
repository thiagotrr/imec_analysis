from __future__ import annotations

from datetime import datetime, timedelta, timezone

from api.models.inspecao_response import InspecaoLaudoResponse
from api.services.historico import listar_historico, persistir_inferencia

from conftest import FakeFirestoreClient


def _response(numero_laudo: str = "2025006988") -> InspecaoLaudoResponse:
    return InspecaoLaudoResponse(
        numero_laudo=numero_laudo,
        classe_prevista="10",
        camada="A",
        resultado="Classe 10 (camada A)",
        resultado_detalhado="Texto detalhado.",
    )


def test_persistir_inferencia_grava_documento_achatado() -> None:
    client = FakeFirestoreClient()

    entry = persistir_inferencia(client, _response(), usuario_id="fulano@energisa.com.br")

    assert entry.id
    assert entry.numero_laudo == "2025006988"
    assert entry.usuario_id == "fulano@energisa.com.br"
    assert entry.criado_em is not None

    stored = client.collection("inferencias").docs[entry.id]
    assert stored["numero_laudo"] == "2025006988"
    assert stored["usuario_id"] == "fulano@energisa.com.br"
    assert "id" not in stored  # id é o Document ID, não um campo gravado


def test_persistir_inferencia_nao_sobrescreve_analises_anteriores() -> None:
    client = FakeFirestoreClient()

    primeira = persistir_inferencia(client, _response(), usuario_id="fulano@energisa.com.br")
    segunda = persistir_inferencia(client, _response(), usuario_id="fulano@energisa.com.br")

    assert primeira.id != segunda.id
    assert len(client.collection("inferencias").docs) == 2


def test_listar_historico_filtra_por_numero_laudo_e_ordena_desc() -> None:
    client = FakeFirestoreClient()
    persistir_inferencia(client, _response("laudo-A"), usuario_id="fulano@energisa.com.br")
    persistir_inferencia(client, _response("laudo-B"), usuario_id="fulano@energisa.com.br")
    persistir_inferencia(client, _response("laudo-A"), usuario_id="fulano@energisa.com.br")

    entries = listar_historico(client, "laudo-A")

    assert len(entries) == 2
    assert all(entry.numero_laudo == "laudo-A" for entry in entries)
    assert entries[0].criado_em >= entries[1].criado_em


def test_listar_historico_sem_inferencias_retorna_lista_vazia() -> None:
    client = FakeFirestoreClient()

    entries = listar_historico(client, "laudo-inexistente")

    assert entries == []


def test_listar_historico_filtra_por_periodo() -> None:
    client = FakeFirestoreClient()
    numero_laudo = "laudo-periodo"
    agora = datetime.now(timezone.utc)

    entry_antiga = persistir_inferencia(client, _response(numero_laudo), usuario_id="fulano@energisa.com.br")
    client.collection("inferencias").docs[entry_antiga.id]["criado_em"] = agora - timedelta(days=10)

    entry_recente = persistir_inferencia(client, _response(numero_laudo), usuario_id="fulano@energisa.com.br")
    client.collection("inferencias").docs[entry_recente.id]["criado_em"] = agora

    entries = listar_historico(client, numero_laudo, data_inicio=agora - timedelta(days=1))

    assert len(entries) == 1
    assert entries[0].id == entry_recente.id
