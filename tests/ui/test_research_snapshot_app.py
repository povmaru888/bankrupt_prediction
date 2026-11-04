"""Deployment checks for the lightweight research snapshot entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "deploy" / "research_snapshot" / "snapshot.json"


def test_snapshot_contract_is_complete_and_small() -> None:
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert len(payload["final_test_comparison"]) == 8
    assert len(payload["training_comparison"]) == 2
    assert len(payload["class_weight_comparison"]) == 4
    assert len(payload["objective_comparison"]) == 3
    assert len(payload["images"]) == 5
    assert len(payload["shap_top_features"]) == 10
    assert SNAPSHOT.stat().st_size < 50_000


def test_lightweight_research_app_renders_without_exceptions() -> None:
    app = AppTest.from_file(str(ROOT / "deploy" / "research_app.py")).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "企業破產預測 · 研究成果"
    assert [header.value for header in app.header] == [
        "XGBoost binary vs Logistic Regression",
        "模型研究比較",
        "評估圖表與 SHAP",
    ]
