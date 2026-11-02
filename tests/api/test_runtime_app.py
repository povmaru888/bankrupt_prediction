from pathlib import Path

from bankruptcy_prediction.api.app import create_runtime_app, load_runtime_settings


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_settings_anchor_service_directories_to_project_root():
    settings = load_runtime_settings(ROOT)
    assert Path(settings.service.models_dir) == (ROOT / "models").resolve()
    assert Path(settings.service.logs_dir) == (ROOT / "logs").resolve()
    assert Path(settings.service.temp_dir) == (ROOT / "tmp" / "uploads").resolve()


def test_runtime_factory_needs_no_arguments():
    app = create_runtime_app()
    assert Path(app.state.settings.service.models_dir).is_absolute()
