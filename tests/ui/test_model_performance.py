from bankruptcy_prediction.ui.pages.model_performance import LIMITATIONS, render_model_performance


def test_model_performance_has_frozen_split_and_mvp_limits():
    calls = []

    class FakeStreamlit:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                return None

            return method

    render_model_performance(
        {
            "version": "v1",
            "average_precision": 0.4,
            "recall_at_top_10_percent": 0.6,
            "mvp_passed": True,
        },
        FakeStreamlit(),
    )
    text = LIMITATIONS + " " + " ".join(str(args) for _, args, _ in calls)
    assert "64%" in text and "16%" in text and "20%" in text
    assert "42" in text and "只評估一次" in text
    assert "2 倍" in text and "50%" in text
