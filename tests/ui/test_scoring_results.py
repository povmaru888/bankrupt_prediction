from bankruptcy_prediction.ui.components.scoring_results import render_scoring_results


def payload(threshold=0.5):
    return {
        "summary": {"threshold_used": threshold},
        "rows": [
            {
                "record_id": "a",
                "row_status": "success",
                "confidence_score": 0.8,
                "risk_class": "high_risk" if threshold <= 0.8 else "not_high_risk",
                "top_10_percent_flag": True,
            },
            {
                "record_id": "b",
                "row_status": "success",
                "confidence_score": 0.2,
                "risk_class": "not_high_risk",
                "top_10_percent_flag": False,
            },
        ],
    }


class FakeStreamlit:
    session_state = {}

    def __init__(self, threshold):
        self.threshold = threshold

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if name == "slider":
                return self.threshold
            if name == "button":
                return False
            return None

        return method


class Client:
    def __init__(self):
        self.calls = []

    def score(self, *args, **kwargs):
        self.calls.append(kwargs)
        return payload(kwargs.get("threshold", 0.5))


def test_threshold_scenario_requeries_api_and_preserves_top_ten():
    client = Client()
    updated = render_scoring_results(
        FakeStreamlit(0.7), client, payload(0.5), csv_bytes=b"x", filename="in.csv"
    )
    assert client.calls[0]["threshold"] == 0.7
    assert updated["rows"][0]["top_10_percent_flag"] is True


def test_empty_success_rows_are_renderable():
    result = render_scoring_results(
        FakeStreamlit(0.5), Client(), {"summary": {"threshold_used": 0.5}, "rows": []}
    )
    assert result["rows"] == []
