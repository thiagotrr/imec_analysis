from __future__ import annotations

import argparse

from main import _resolve_api_run_config


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "https": None,
        "host": None,
        "port": None,
        "reload": None,
        "ssl_certfile": None,
        "ssl_keyfile": None,
        "ssl_keyfile_password": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_paas_port_env_is_honored(monkeypatch: object) -> None:
    monkeypatch.delenv("IMEC_API_PORT", raising=False)
    monkeypatch.setenv("PORT", "8080")
    config = _resolve_api_run_config(_args())
    assert config["port"] == 8080
    assert config["reload"] is False


def test_imec_api_port_takes_precedence_over_port(monkeypatch: object) -> None:
    monkeypatch.setenv("IMEC_API_PORT", "9000")
    monkeypatch.setenv("PORT", "8080")
    config = _resolve_api_run_config(_args())
    assert config["port"] == 9000
