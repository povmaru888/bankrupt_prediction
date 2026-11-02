"""Test-only end-to-end training workflow and CLI failure boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from bankruptcy_prediction.cli import main
from bankruptcy_prediction.settings import load_settings
from bankruptcy_prediction.training import orchestrator
from bankruptcy_prediction.training.logistic_pipeline import build_logistic_pipeline
from bankruptcy_prediction.training.model_search import ModelSearchResult, SearchResults
from bankruptcy_prediction.training.orchestrator import run_training

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CSV = ROOT / "Database" / "bankrupt_data.csv"


def _small_csv(path: Path, rows: int = 80) -> Path:
    columns = [name.strip() for name in pd.read_csv(OFFICIAL_CSV, nrows=0).columns]
    feature_names = [name for name in columns if name != "Bankrupt?"]
    rng = np.random.default_rng(42)
    labels = np.array([0, 1] * (rows // 2), dtype=int)
    if rows % 2:
        labels = np.append(labels, 0)
    frame = pd.DataFrame(rng.normal(size=(len(labels), len(feature_names))), columns=feature_names)
    frame[feature_names[0]] = labels * 2.0 + rng.normal(scale=0.2, size=len(labels))
    frame["Net Income Flag"] = 0
    frame["Current Liability to Liability"] = frame["Current Liabilities/Liability"]
    frame["Current Liability to Equity"] = frame["Current Liabilities/Equity"]
    frame["Bankrupt?"] = labels
    frame.to_csv(path, index=False)
    return path


def _quick_config(path: Path) -> Path:
    baseline = yaml.safe_load((ROOT / "config" / "training.yaml").read_text(encoding="utf-8"))
    baseline["test_only"] = True
    baseline["search"].update(n_iter=1, cv_folds=2)
    baseline["calibration"].update(cv_folds=2)
    path.write_text(yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8")
    return path


def _settings(path: Path):
    return load_settings(path, ROOT / "config" / "service.yaml").training


def test_training_normalizes_source_headers_before_search(tmp_path, monkeypatch):
    source = _small_csv(tmp_path / "spaced.csv")
    frame = pd.read_csv(source)
    frame.columns = [name if name == "Bankrupt?" else f" {name} " for name in frame.columns]
    frame.to_csv(source, index=False)

    def inspect_search(X_train, y_train, config, *, test_only):
        assert all(name == name.strip() for name in X_train.columns)
        assert "Net Income Flag" in X_train.columns
        raise RuntimeError("normalized-input-verified")

    monkeypatch.setattr(orchestrator, "search_models", inspect_search)
    with pytest.raises(RuntimeError, match="normalized-input-verified"):
        run_training(source, _settings(_quick_config(tmp_path / "quick.yaml")), tmp_path / "models")


def _quick_search(X_train: Any, y_train: Any, config: Any, *, test_only: bool) -> SearchResults:
    assert test_only and config.test_only
    estimator = build_logistic_pipeline(max_iter=250)
    logistic = ModelSearchResult(
        model_name="logistic_regression",
        attempted_candidates=config.search.n_iter,
        best_score=0.5,
        best_parameters={"classifier__C": 1.0},
        best_estimator=estimator,
    )
    xgboost = ModelSearchResult(
        model_name="xgboost",
        attempted_candidates=config.search.n_iter,
        best_score=0.5,
        best_parameters={"test_double": True},
        best_estimator=estimator,
    )
    return SearchResults(logistic_regression=logistic, xgboost=xgboost)


def test_test_only_cli_workflow_builds_package_once_and_preserves_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = _small_csv(tmp_path / "small.csv")
    config_path = _quick_config(tmp_path / "quick.yaml")
    models_root = tmp_path / "models"
    models_root.mkdir()
    active = models_root / "active.json"
    active.write_text('{"version":"keep-me"}', encoding="utf-8")
    monkeypatch.setattr(orchestrator, "search_models", _quick_search)

    result = run_training(data_path, _settings(config_path), models_root)

    assert result.test_only is True
    assert result.quality_gate_passed is False
    assert result.to_dict()["status"] == "test_only_completed"
    assert result.package.path.is_dir()
    manifest = json.loads(result.package.manifest_path.read_text(encoding="utf-8"))
    assert manifest["test_only"] is True
    assert manifest["model_family"] == "logistic_regression"
    assert manifest["threshold"]["default_threshold"] == result.metrics.threshold
    assert result.package.path.joinpath("model_report.html").is_file()
    assert "test-only 測試用途" in result.package.report_path.read_text(encoding="utf-8")
    assert active.read_text(encoding="utf-8") == '{"version":"keep-me"}'
    assert len(list(models_root.glob("candidate-*"))) == 1
    assert not list(models_root.glob(".candidate-staging-*"))


def test_failure_before_package_publication_leaves_no_incomplete_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = _small_csv(tmp_path / "small.csv")
    config_path = _quick_config(tmp_path / "quick.yaml")
    models_root = tmp_path / "models"
    models_root.mkdir()
    active = models_root / "active.json"
    active.write_text('{"version":"active-v1"}', encoding="utf-8")

    def fail_search(*args: Any, **kwargs: Any) -> SearchResults:
        raise RuntimeError("synthetic search failure")

    monkeypatch.setattr(orchestrator, "search_models", fail_search)
    with pytest.raises(RuntimeError, match="synthetic search failure"):
        run_training(data_path, _settings(config_path), models_root)

    assert active.read_text(encoding="utf-8") == '{"version":"active-v1"}'
    assert list(models_root.iterdir()) == [active]


def test_cli_prints_structured_training_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _quick_config(tmp_path / "quick.yaml")
    data_path = tmp_path / "data.csv"
    data_path.write_text("placeholder", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Outcome:
        def to_dict(self) -> dict[str, object]:
            return {"status": "quality_gate_failed", "test_only": True}

    def fake_run(data: str, config: Any, root: str) -> _Outcome:
        captured.update(data=data, config=config, root=root)
        return _Outcome()

    monkeypatch.setattr("bankruptcy_prediction.cli.run_training", fake_run)

    exit_code = main(
        [
            "train",
            "--data",
            str(data_path),
            "--config",
            str(config_path),
            "--models-root",
            str(tmp_path / "models"),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output == {"status": "quality_gate_failed", "test_only": True}
    assert captured["config"].test_only is True
    assert captured["data"] == str(data_path)


def test_quick_settings_are_rejected_without_test_only_marker(tmp_path: Path) -> None:
    config_path = _quick_config(tmp_path / "quick.yaml")
    settings = _settings(config_path)

    with pytest.raises(ValueError, match="requires 30 search candidates"):
        run_training("missing.csv", replace(settings, test_only=False), tmp_path / "models")
