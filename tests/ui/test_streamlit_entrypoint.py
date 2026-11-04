from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def test_root_entrypoint_serves_lightweight_research_snapshot_by_default():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "企業破產預測 · 研究成果"
    assert len(app.radio) == 0
