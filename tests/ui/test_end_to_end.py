from bankruptcy_prediction.ui.session_state import (
    clear_score_session,
    get_score_session,
    save_score_session,
)


def test_score_session_is_shared_and_clearable_without_disk_history():
    state = {}
    payload = {"summary": {"model_version": "v1"}, "rows": []}
    save_score_session(state, payload, b"f1\n1\n", "input.csv")
    assert get_score_session(state) == (payload, b"f1\n1\n", "input.csv")
    clear_score_session(state)
    assert get_score_session(state) is None
    assert state == {}
