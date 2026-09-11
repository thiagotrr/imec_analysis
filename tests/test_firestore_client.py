from __future__ import annotations

import pytest

from db.firestore.client import FirestoreRuntimeError, get_collection, load_firestore_runtime
from db.firestore.settings import FirestoreSettings, load_firestore_settings

from conftest import FakeFirestoreClient, FakeFirestoreRuntime


def test_load_firestore_settings_defaults_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # tests/conftest.py força FIRESTORE_ENABLED=false para toda a suíte.
    settings = load_firestore_settings()
    assert settings.enabled is False
    assert settings.database_id == "imec-analysis"


def test_load_firestore_settings_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRESTORE_ENABLED", "true")
    monkeypatch.setenv("FIRESTORE_PROJECT_ID", "meu-projeto")
    monkeypatch.setenv("FIRESTORE_DATABASE_ID", "outro_banco")

    settings = load_firestore_settings()

    assert settings.enabled is True
    assert settings.project_id == "meu-projeto"
    assert settings.database_id == "outro_banco"


def test_load_firestore_runtime_raises_when_disabled() -> None:
    with pytest.raises(FirestoreRuntimeError):
        load_firestore_runtime()


def test_get_collection_delegates_to_client() -> None:
    fake_client = FakeFirestoreClient()
    runtime = FakeFirestoreRuntime(client=fake_client)

    collection = get_collection(runtime, "minha_colecao")

    assert collection is fake_client.collection("minha_colecao")


def test_firestore_settings_is_frozen_dataclass() -> None:
    settings = FirestoreSettings()
    with pytest.raises(Exception):
        settings.enabled = True  # type: ignore[misc]
