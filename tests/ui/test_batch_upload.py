from bankruptcy_prediction.ui.pages.batch_scoring import UPLOAD_RULES, render_batch_scoring


class FakeStreamlit:
    def __init__(self, uploaded=None, *, clicked=True):
        self.uploaded = uploaded
        self.clicked = clicked
        self.session_state = {}
        self.messages = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.messages.append((name, args, kwargs))
            if name == "file_uploader":
                return self.uploaded
            if name == "slider":
                return kwargs.get("value", 0.5)
            if name == "button":
                return self.uploaded is not None and self.clicked
            return None

        return method


class Upload:
    name = "input.csv"

    def getvalue(self):
        return b"f1\n1\n"


class Client:
    def __init__(self):
        self.called = False
        self.threshold = None

    def score(self, *args, **kwargs):
        self.called = True
        self.threshold = kwargs.get("threshold")
        return {"summary": {"total_rows": 1, "succeeded_rows": 1, "failed_rows": 0}}


def test_upload_rules_and_success_summary():
    st = FakeStreamlit(Upload())
    client = Client()
    payload = render_batch_scoring(st, client, model_loaded=True)
    assert "10 MB" in UPLOAD_RULES and "10,000" in UPLOAD_RULES and "95" in UPLOAD_RULES
    assert client.called is True
    assert payload["summary"]["succeeded_rows"] == 1


def test_no_model_disables_upload():
    st = FakeStreamlit(Upload())
    assert render_batch_scoring(st, Client(), model_loaded=False) is None
    assert any(name == "warning" for name, _, _ in st.messages)


def test_active_model_threshold_is_initial_upload_threshold():
    st = FakeStreamlit(Upload())
    client = Client()
    render_batch_scoring(st, client, model_loaded=True, default_threshold=0.049794)
    assert client.threshold == 0.049794


def test_selecting_new_unscored_file_clears_previous_result():
    st = FakeStreamlit(Upload(), clicked=False)
    st.session_state.update(
        {
            "score_response": {"rows": []},
            "uploaded_csv_bytes": b"old\n1\n",
            "uploaded_filename": "old.csv",
        }
    )
    assert render_batch_scoring(st, Client(), model_loaded=True) is None
    assert "score_response" not in st.session_state
    assert any(name == "info" for name, _, _ in st.messages)
