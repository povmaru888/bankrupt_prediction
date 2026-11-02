from bankruptcy_prediction.ui.pages.case_explanation import render_case_explanation


class FakeStreamlit:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "selectbox":
                return "row_000001"
            return None

        return method


def test_case_page_shows_values_and_non_causal_explanation():
    payload = {
        "rows": [
            {
                "record_id": "row_000001",
                "row_status": "success",
                "model_version": "v1",
                "confidence_score": 0.8,
                "risk_class": "high_risk",
                "positive_factors": [{"feature_name": "f1", "contribution": 0.2}],
                "negative_factors": [{"feature_name": "f2", "contribution": -0.1}],
            }
        ]
    }
    st = FakeStreamlit()
    render_case_explanation(st, payload, b"f1,f2\n1,2\n")
    text = " ".join(str(args) for _, args, _ in st.calls)
    assert "原始輸入值" in text
    assert "因果" in text
    assert "row_000001" in text


def test_case_page_handles_empty_success_rows():
    st = FakeStreamlit()
    render_case_explanation(st, {"rows": []})
    assert any(name == "info" for name, _, _ in st.calls)
