"""Ensure scoring audit logs contain aggregate metadata only."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bankruptcy_prediction.logging_config import configure_logging, log_score_summary
from bankruptcy_prediction.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_scoring_log_uses_allowlist_and_does_not_leak_sensitive_values(
    tmp_path: Path,
) -> None:
    settings = load_settings(ROOT / "config/training.yaml", ROOT / "config/service.yaml")
    log_dir = tmp_path / "audit"
    settings = replace(settings, service=replace(settings.service, logs_dir=str(log_dir)))
    configure_logging(settings)

    request_id = log_score_summary(
        model_version="model-v001",
        total_rows=2,
        succeeded_rows=1,
        failed_rows=1,
        elapsed_ms=245,
        error_code_counts={"INVALID_ROW": 1},
        request_id="request-001",
    )
    assert request_id == "request-001"

    log_file = log_dir / "scoring_audit.jsonl"
    entry = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert set(entry) == {
        "timestamp",
        "request_id",
        "model_version",
        "total_rows",
        "succeeded_rows",
        "failed_rows",
        "elapsed_ms",
        "error_code_counts",
    }
    output = log_file.read_text(encoding="utf-8")
    for sensitive_value in (
        "sensitive-company-name",
        "C:\\private\\input.csv",
        "record_very_sensitive_42",
        "0.987654",
        "cash_ratio_contribution",
    ):
        assert sensitive_value not in output


@pytest.mark.parametrize("model_version", ["C:\\private\\model.joblib", "../secret"])
def test_rejects_path_like_model_version(model_version: str, tmp_path: Path) -> None:
    settings = load_settings(ROOT / "config/training.yaml", ROOT / "config/service.yaml")
    settings = replace(settings, service=replace(settings.service, logs_dir=str(tmp_path)))
    configure_logging(settings)

    with pytest.raises(ValueError, match="model_version"):
        log_score_summary(
            model_version=model_version,
            total_rows=0,
            succeeded_rows=0,
            failed_rows=0,
            elapsed_ms=0,
            error_code_counts={},
        )


def test_invalid_counts_are_rejected(tmp_path: Path) -> None:
    settings = load_settings(ROOT / "config/training.yaml", ROOT / "config/service.yaml")
    settings = replace(settings, service=replace(settings.service, logs_dir=str(tmp_path)))
    configure_logging(settings)

    with pytest.raises(ValueError, match="must equal total_rows"):
        log_score_summary(
            model_version="model-v001",
            total_rows=2,
            succeeded_rows=1,
            failed_rows=0,
            elapsed_ms=0,
            error_code_counts={},
        )
