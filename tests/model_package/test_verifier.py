"""Tests for safe verification order, schema checks, and artifact checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bankruptcy_prediction.model_package.builder import (
    CandidatePackageContext,
    build_candidate_package,
)
from bankruptcy_prediction.model_package.verifier import (
    PackageVerificationError,
    verify_package,
)
from bankruptcy_prediction.reporting.model_report import ModelReportContext


class _PickleableModel:
    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile([0.75, 0.25], (len(X), 1))


def _create_package(root: Path) -> Path:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = {"threshold": 0.5, "average_precision": 1.0}
    report = ModelReportContext(
        dataset_audit={"rows": 4},
        split_summary={"seed": 42},
        search_summary={"n_iter": 30},
        selection_summary={"model_family": "test_model"},
        threshold_summary={"default_threshold": 0.5},
        metrics_report=metrics,
        y_test=labels,
        confidence_scores=scores,
        mean_abs_shap={"a": 0.2},
        model_family="test_model",
    )
    context = CandidatePackageContext(
        model=_PickleableModel(),
        model_family="test_model",
        data_sha256=hashlib.sha256(b"data").hexdigest(),
        split_seed=42,
        split_indices={"train": [0, 1], "validation": [2], "test": [3]},
        feature_names=("a", "b"),
        removed_columns=(),
        package_versions={"python": "3.12"},
        model_parameters={},
        calibration={"method": "sigmoid"},
        threshold_result={"default_threshold": 0.5, "rule": "lowest"},
        metrics_report=metrics,
        report_context=report,
    )
    return build_candidate_package(context, root).path


def test_verifies_complete_package_without_loading_joblib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_path = _create_package(tmp_path / "models")
    import joblib

    monkeypatch.setattr(
        joblib,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not deserialize")),
    )

    package = verify_package(package_path)

    assert package.version == package_path.name
    assert package.path == package_path.resolve()
    assert package.feature_schema["feature_names"] == ["a", "b"]
    assert package.metrics["threshold"] == 0.5
    assert package.model_path.is_file()


@pytest.mark.parametrize("damage", ["missing", "tampered", "unknown_schema", "manifest_incomplete"])
def test_rejects_missing_tampered_or_incompatible_artifacts(tmp_path: Path, damage: str) -> None:
    package = _create_package(tmp_path / "models")
    if damage == "missing":
        (package / "charts" / "calibration.png").unlink()
    elif damage == "tampered":
        (package / "metrics.json").write_text("{}", encoding="utf-8")
    elif damage == "unknown_schema":
        schema_path = package / "feature_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["schema_version"] = 999
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        manifest_path = package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["sha256"]["feature_schema.json"] = hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        manifest_path = package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["calibration"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackageVerificationError):
        verify_package(package)


def test_rejects_checksum_path_traversal_and_does_not_echo_absolute_path(
    tmp_path: Path,
) -> None:
    package = _create_package(tmp_path / "models")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["sha256"]["../../outside.joblib"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackageVerificationError) as error:
        verify_package(package)

    assert str(package) not in str(error.value)
    assert ".." not in str(error.value)
