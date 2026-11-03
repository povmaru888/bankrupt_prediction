import pytest

from bankruptcy_prediction.ui.api_client import ApiClient, ApiClientError
from bankruptcy_prediction.ui.app import PAGE_NAMES, render_app


def test_client_rejects_unavailable_api(monkeypatch):
    import bankruptcy_prediction.ui.api_client as module

    def fail(*args, **kwargs):
        raise module.httpx.ConnectError("offline")

    monkeypatch.setattr(module.httpx, "request", fail)
    with pytest.raises(ApiClientError, match="無法連線"):
        ApiClient().health()


def test_app_shell_has_four_pages_and_disables_scoring_without_model():
    calls = []

    class FakeStreamlit:
        session_state = {}

        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                if name == "radio":
                    return PAGE_NAMES[1]
                return None

            return method

    class OfflineClient:
        def health(self):
            raise ApiClientError("offline")

    render_app(OfflineClient(), streamlit_module=FakeStreamlit())
    assert len(PAGE_NAMES) == 4
    button = next(call for call in calls if call[0] == "button")
    assert button[2]["disabled"] is True


def test_app_shell_displays_safe_model_reason_code():
    calls = []

    class FakeStreamlit:
        session_state = {}

        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                if name == "radio":
                    return PAGE_NAMES[1]
                return None

            return method

    class DegradedClient:
        def health(self):
            return {
                "status": "degraded",
                "active_model_loaded": False,
                "reason_code": "model_load_failed",
            }

    render_app(DegradedClient(), streamlit_module=FakeStreamlit())
    warning = next(call for call in calls if call[0] == "warning")
    assert "reason_code: model_load_failed" in warning[1][0]
