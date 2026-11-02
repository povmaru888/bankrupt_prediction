"""CLI model lifecycle command coverage without touching the production models root."""

import json

from bankruptcy_prediction.cli import build_parser, main


def test_cli_exposes_train_activate_and_rollback_commands():
    parser = build_parser()
    assert parser.parse_args(["activate", "--version", "v1"]).command == "activate"
    assert parser.parse_args(["rollback", "--version", "v1"]).command == "rollback"


def test_unknown_version_failure_does_not_create_or_replace_active(tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    active = models / "active.json"
    active.write_text(json.dumps({"schema_version": 1, "version": "keep"}), encoding="utf-8")
    exit_code = main(["activate", "--version", "missing", "--models-root", str(models)])
    assert exit_code == 1
    assert json.loads(active.read_text(encoding="utf-8"))["version"] == "keep"
    assert "failed" in capsys.readouterr().err
