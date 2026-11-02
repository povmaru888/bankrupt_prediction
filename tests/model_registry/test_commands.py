"""Tests for validated activation, rollback, and atomic active pointer updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bankruptcy_prediction.cli import main
from bankruptcy_prediction.model_package.builder import (
    CandidatePackageContext,
    build_candidate_package,
)
from bankruptcy_prediction.model_registry import commands
from bankruptcy_prediction.model_registry.commands import (
    RegistryCommandError,
    activate_model,
    rollback_model,
)
from bankruptcy_prediction.reporting.model_report import ModelReportContext


class _Model:
    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile([0.75, 0.25], (len(X), 1))


def _package(models_root: Path, *, report_family: str = "test_model") -> str:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = {"threshold": 0.5, "average_precision": 1.0}
    context = CandidatePackageContext(
        model=_Model(),
        model_family=report_family,
        data_sha256=hashlib.sha256(report_family.encode()).hexdigest(),
        split_seed=42,
        split_indices={"train": [0, 1], "validation": [2], "test": [3]},
        feature_names=("a", "b"),
        removed_columns=(),
        package_versions={"python": "3.12"},
        model_parameters={},
        calibration={"method": "sigmoid"},
        threshold_result={"default_threshold": 0.5, "rule": "lowest"},
        metrics_report=metrics,
        report_context=ModelReportContext(
            dataset_audit={"rows": 4},
            split_summary={"seed": 42},
            search_summary={"n_iter": 30},
            selection_summary={"model_family": report_family},
            threshold_summary={"default_threshold": 0.5},
            metrics_report=metrics,
            y_test=labels,
            confidence_scores=scores,
            mean_abs_shap={"a": 0.2},
            model_family=report_family,
        ),
    )
    return build_candidate_package(context, models_root).version


def test_t09_activate_and_rollback_switch_versions_without_deleting(tmp_path: Path) -> None:
    version_a = _package(tmp_path)
    version_b = _package(tmp_path, report_family="other_model")

    first = activate_model(version_a, tmp_path)
    second = activate_model(version_b, tmp_path)
    rollback = rollback_model(version_a, tmp_path)
    pointer = json.loads((tmp_path / "active.json").read_text(encoding="utf-8"))

    assert first.previous_version is None
    assert second.previous_version == version_a
    assert rollback.action == "rollback"
    assert rollback.previous_version == version_b
    assert pointer["version"] == version_a
    assert pointer["previous_version"] == version_b
    assert (tmp_path / version_a).is_dir() and (tmp_path / version_b).is_dir()


def test_invalid_or_corrupt_version_preserves_active_pointer(tmp_path: Path) -> None:
    active_version = _package(tmp_path)
    broken_version = _package(tmp_path, report_family="broken_candidate")
    activate_model(active_version, tmp_path)
    active_path = tmp_path / "active.json"
    previous = active_path.read_bytes()
    (tmp_path / broken_version / "model.joblib").write_bytes(b"damaged")

    with pytest.raises(RegistryCommandError, match="failed package verification"):
        activate_model(broken_version, tmp_path)
    with pytest.raises(RegistryCommandError, match="unavailable"):
        rollback_model("missing-version", tmp_path)

    assert active_path.read_bytes() == previous


def test_interrupted_atomic_write_keeps_previous_pointer_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version = _package(tmp_path)
    active_path = tmp_path / "active.json"
    active_path.write_text('{"version":"previous"}', encoding="utf-8")
    previous = active_path.read_bytes()

    def interrupted_replace(source: object, destination: object) -> None:
        raise OSError("synthetic interruption")

    monkeypatch.setattr(commands.os, "replace", interrupted_replace)
    with pytest.raises(RegistryCommandError, match="previous value preserved"):
        activate_model(version, tmp_path)

    assert active_path.read_bytes() == previous
    assert not list(tmp_path.glob(".active-*.tmp"))


def test_rejects_path_like_version_ids(tmp_path: Path) -> None:
    with pytest.raises(RegistryCommandError, match="identifier is invalid"):
        activate_model("..\\outside", tmp_path)


def test_cli_exposes_activate_and_rollback_commands(monkeypatch, capsys) -> None:
    class _Result:
        def to_dict(self) -> dict[str, str]:
            return {"action": "activate", "version": "candidate-v1"}

    monkeypatch.setattr("bankruptcy_prediction.cli.activate_model", lambda version, root: _Result())
    exit_code = main(["activate", "--version", "candidate-v1", "--models-root", "models"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "action": "activate",
        "version": "candidate-v1",
    }
