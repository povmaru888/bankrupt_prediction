from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]


def test_root_entrypoint_executes_without_import_errors_or_auto_pages():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
    assert not app.exception
    assert len(app.title) == 1
    assert len(app.radio) == 1
    assert len(app.radio[0].options) == 4
