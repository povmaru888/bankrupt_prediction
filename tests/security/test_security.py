from pathlib import Path

import pytest

from bankruptcy_prediction.contracts import BatchSummary, RowStatus
from bankruptcy_prediction.export.csv_exporter import export_results_csv
from bankruptcy_prediction.scoring.result_models import RiskClass, ScoringResult, ScoringResultRow
from bankruptcy_prediction.service.temp_files import (
    create_upload_file_path,
    temporary_upload_workspace,
)
from bankruptcy_prediction.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_service_is_loopback_only_and_upload_paths_ignore_traversal(tmp_path):
    settings = load_settings(ROOT / "config/training.yaml", ROOT / "config/service.yaml")
    assert settings.service.host == "127.0.0.1"
    with temporary_upload_workspace(tmp_path) as workspace:
        path = create_upload_file_path(workspace, "..\\..\\evil.csv")
        assert path.parent == workspace.resolve()
        assert path.name.startswith("upload-")


def test_managed_upload_workspace_cleans_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with temporary_upload_workspace(tmp_path) as workspace:
            create_upload_file_path(workspace)
            raise RuntimeError("synthetic interruption")
    assert list(tmp_path.iterdir()) == []


def test_formula_injection_is_escaped_in_export():
    row = ScoringResultRow(
        record_id="=SUM(1,2)",
        model_version="v1",
        row_status=RowStatus.SUCCESS,
        confidence_score=0.5,
        batch_percentile=100,
        threshold_used=0.5,
        risk_class=RiskClass.HIGH_RISK,
        top_10_percent_flag=True,
        positive_factors=(),
        negative_factors=(),
    )
    exported = export_results_csv(ScoringResult(BatchSummary(1, 1, 0, "v1", 0.5), (row,)))
    assert b"'=SUM(1,2)" in exported
