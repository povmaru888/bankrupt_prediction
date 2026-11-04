from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launch_scripts_are_loopback_only_and_use_expected_ports():
    api = (ROOT / "scripts" / "start_api.ps1").read_text(encoding="utf-8")
    ui = (ROOT / "scripts" / "start_ui.ps1").read_text(encoding="utf-8")
    assert "127.0.0.1" in api and "--port $Port" in api
    assert "127.0.0.1" in ui and "8502" in ui and "8000" in ui
    assert "0.0.0.0" not in api + ui


def test_ui_launcher_uses_stable_root_entrypoint_and_src_import_path():
    ui = (ROOT / "scripts" / "start_ui.ps1").read_text(encoding="utf-8")
    assert '$env:PYTHONPATH = (Join-Path $ProjectRoot "src")' in ui
    assert '$env:BANKRUPTCY_FULL_LOCAL_APP = "1"' in ui
    assert 'Join-Path $ProjectRoot "streamlit_app.py"' in ui
    assert "src\\bankruptcy_prediction\\ui\\app.py" not in ui
    assert (ROOT / "streamlit_app.py").is_file()


def test_api_launcher_uses_zero_argument_runtime_factory_and_src_import_path():
    api = (ROOT / "scripts" / "start_api.ps1").read_text(encoding="utf-8")
    assert '$env:PYTHONPATH = (Join-Path $ProjectRoot "src")' in api
    assert "bankruptcy_prediction.api.app:create_runtime_app" in api
    assert "bankruptcy_prediction.api.app:create_app --factory" not in api


def test_runbook_covers_lifecycle_and_failures():
    runbook = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    for phrase in ("無 active model", "checksum", "rollback", "暫存", "127.0.0.1"):
        assert phrase in runbook
