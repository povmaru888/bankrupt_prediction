from bankruptcy_prediction.ui.pages.batch_scoring import UPLOAD_RULES, render_batch_scoring


class FakeStreamlit:
    session_state = {}

    def __init__(self, uploaded=None):
        self.uploaded = uploaded
        self.messages = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.messages.append((name, args, kwargs))
            if name == "file_uploader":
                return self.uploaded
            if name == "slider":
                return 0.5
            if name == "button":
                return self.uploaded is not None
            return None

        return method


class Upload:
    name = "input.csv"

    def getvalue(self):
        return b"f1\n1\n"


class Client:
    def __init__(self):
        self.called = False

    def score(self, *args, **kwargs):
        self.called = True
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
