"""Tests for candidate model-package integrity and no-overwrite behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bankruptcy_prediction.model_package import builder
from bankruptcy_prediction.model_package.builder import (
    CandidatePackageContext,
    build_candidate_package,
)
from bankruptcy_prediction.reporting.model_report import ModelReportContext


class _PickleableModel:
    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile([0.75, 0.25], (len(X), 1))


def _context() -> CandidatePackageContext:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    report = ModelReportContext(
        dataset_audit={"rows": 4, "features": 2},
        split_summary={"seed": 42},
        search_summary={"n_iter": 30},
        selection_summary={"model_family": "test_model"},
        threshold_summary={"default_threshold": 0.5},
        metrics_report={"threshold": 0.5, "average_precision": 1.0},
        y_test=labels,
        confidence_scores=scores,
        mean_abs_shap={"a": 0.2},
        model_family="test_model",
    )
    return CandidatePackageContext(
        model=_PickleableModel(),
        model_family="test_model",
        data_sha256=hashlib.sha256(b"data").hexdigest(),
        split_seed=42,
        split_indices={
            "train": [0, 1, 2, 3],
            "validation": [4, 5],
            "test": [6, 7],
        },
        feature_names=("a", "b"),
        removed_columns=("Net Income Flag",),
        package_versions={"scikit-learn": "1.9.1"},
        model_parameters={"C": 1.0},
        calibration={"method": "sigmoid", "cv_folds": 5},
        threshold_result={"default_threshold": 0.5, "rule": "lowest"},
        metrics_report={"threshold": 0.5, "average_precision": 1.0},
        report_context=report,
    )


def test_builds_complete_candidate_with_valid_checksums(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    active_path = models_root / "active.json"
    active_path.write_text('{"version":"previous-active"}', encoding="utf-8")
    package = build_candidate_package(_context(), models_root)

    required = (
        "model.joblib",
        "feature_schema.json",
        "metrics.json",
        "model_report.html",
        "charts/precision_recall.png",
        "charts/calibration.png",
        "charts/confusion_matrix.png",
        "charts/feature_importance.png",
    )
    manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    for name in required:
        artifact = package.path / name
        assert artifact.is_file()
        assert (
            manifest["files"]["sha256"][name] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
    assert manifest["status"] == "candidate"
    assert manifest["data_sha256"] == _context().data_sha256
    assert manifest["split"]["seed"] == 42
    assert manifest["split"]["partitions"]["validation"]["row_count"] == 2
    assert manifest["feature_names"] == ["a", "b"]
    assert manifest["removed_columns"] == ["Net Income Flag"]
    assert manifest["threshold"]["default_threshold"] == 0.5
    assert active_path.read_text(encoding="utf-8") == '{"version":"previous-active"}'
    assert not list(package.path.parent.glob(".candidate-staging-*"))


def test_does_not_overwrite_existing_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    version = "candidate-fixed-version"
    destination = root / version
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(builder, "_version_id", lambda: version)

    with pytest.raises(FileExistsError):
        build_candidate_package(_context(), root)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not list(root.glob(".candidate-staging-*"))


@pytest.mark.parametrize(
    "changes",
    [
        {"data_sha256": "not-a-hash"},
        {"split_indices": {"train": [0], "validation": [0], "test": [1]}},
        {"feature_names": ("a", "a")},
        {"threshold_result": {"default_threshold": 1.2}},
    ],
)
def test_invalid_context_never_publishes_package(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        build_candidate_package(replace(_context(), **changes), tmp_path / "models")
    assert not (tmp_path / "models").exists()
