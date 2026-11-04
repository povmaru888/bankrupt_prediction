from __future__ import annotations

from types import SimpleNamespace

from bankruptcy_prediction.ui import embedded_api


def test_reuses_existing_loopback_api(monkeypatch) -> None:
    monkeypatch.setattr(embedded_api, "_INSTANCE", None)
    monkeypatch.setattr(embedded_api, "_port_accepts_connections", lambda host, port: True)

    assert embedded_api.ensure_embedded_api() is None


def test_reuses_live_embedded_instance(monkeypatch) -> None:
    instance = SimpleNamespace(
        thread=SimpleNamespace(is_alive=lambda: True),
        host="127.0.0.1",
        port=8000,
    )
    monkeypatch.setattr(embedded_api, "_INSTANCE", instance)

    assert embedded_api.ensure_embedded_api() is instance
