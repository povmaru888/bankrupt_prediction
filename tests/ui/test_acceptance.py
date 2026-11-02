from pathlib import Path

from bankruptcy_prediction.ui.app import PAGE_NAMES
from bankruptcy_prediction.ui.components.scoring_results import render_scoring_results

ROOT = Path(__file__).resolve().parents[2]


def test_all_four_pages_and_no_forbidden_probability_language():
    assert PAGE_NAMES == ("總覽與模型資訊", "批次評分", "個案解釋", "模型效能")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "bankruptcy_prediction" / "ui").rglob("*.py")
    )
    for forbidden in ("破產機率", "將會破產", "不會破產"):
        assert forbidden not in text


def test_top_ten_flag_is_displayed_from_api_response_not_threshold_recomputed():
    payload = {
        "summary": {"threshold_used": 0.5},
        "rows": [
            {
                "record_id": "a",
                "row_status": "success",
                "confidence_score": 0.4,
                "risk_class": "not_high_risk",
                "top_10_percent_flag": True,
            },
        ],
    }

    class St:
        session_state = {}

        def __getattr__(self, name):
            def method(*args, **kwargs):
                if name == "slider":
                    return 0.5
                return None

            return method

    result = render_scoring_results(St(), object(), payload)
    assert result["rows"][0]["top_10_percent_flag"] is True
