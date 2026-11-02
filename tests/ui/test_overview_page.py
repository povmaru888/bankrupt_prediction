from bankruptcy_prediction.ui.pages.overview import LIMITATION_TEXT, render_overview


def test_overview_contains_required_limits_and_metrics():
    calls = []

    class FakeStreamlit:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                if name == "columns":
                    return [self, self, self]
                return None

            return method

    render_overview(
        {
            "version": "v1",
            "model_family": "logistic_regression",
            "created_at": "today",
            "default_threshold": 0.5,
            "average_precision": 0.4,
            "recall_at_top_10_percent": 0.6,
        },
        FakeStreamlit(),
    )
    text = " ".join(str(args) for _, args, _ in calls)
    assert "公司" in LIMITATION_TEXT and "年度" in LIMITATION_TEXT and "預測窗口" in LIMITATION_TEXT
    assert "分析優先順序" in text
    assert "破產機率" not in text
