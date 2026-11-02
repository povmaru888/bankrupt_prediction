"""Tests for read-only active-model loading, error states, and caching."""

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
from bankruptcy_prediction.model_registry.commands import activate_model
from bankruptcy_prediction.model_registry.registry import (
    ActiveModelUnavailable,
    load_active_model,
)
from bankruptcy_prediction.reporting.model_report import ModelReportContext


class _Model:
    def predict_proba(self, X: object) -> np.ndarray:
        return np.tile([0.75, 0.25], (len(X), 1))


def _make_package(root: Path) -> str:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = {
        "threshold": 0.5,
        "average_precision": 0.9,
        "recall_at_top_10_percent": 0.5,
        "mvp_passed": True,
    }
    context = CandidatePackageContext(
        model=_Model(),
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
        report_context=ModelReportContext(
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
        ),
    )
    return build_candidate_package(context, root).version


def test_loads_active_model_and_returns_read_only_summary(tmp_path: Path) -> None:
    version = _make_package(tmp_path)
    activate_model(version, tmp_path)

    active = load_active_model(tmp_path)

    assert active.version == version
    assert active.model.predict_proba(np.zeros((2, 2))).shape == (2, 2)
    assert active.feature_names == ("a", "b")
    assert active.default_threshold == 0.5
    assert active.summary.average_precision == 0.9
    assert active.summary.to_dict()["feature_count"] == 2
    with pytest.raises(TypeError):
        active.metrics["new"] = 1  # type: ignore[index]
    assert not hasattr(active, "activate") and not hasattr(active, "rollback")


@pytest.mark.parametrize(
    "setup,error_code",
    [
        ("none", "no_active_model"),
        ("missing", "invalid_active_model"),
        ("damaged", "invalid_active_model"),
        ("unknown_schema", "invalid_active_model"),
    ],
)
def test_t10_reports_unavailable_causes(tmp_path: Path, setup: str, error_code: str) -> None:
    if setup == "missing":
        (tmp_path / "active.json").write_text(
            json.dumps({"schema_version": 1, "version": "missing-version"}), encoding="utf-8"
        )
    elif setup == "damaged":
        version = _make_package(tmp_path)
        activate_model(version, tmp_path)
        (tmp_path / version / "model.joblib").write_bytes(b"corrupt")
    elif setup == "unknown_schema":
        version = _make_package(tmp_path)
        activate_model(version, tmp_path)
        schema_path = tmp_path / version / "feature_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["schema_version"] = 99
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        manifest_path = tmp_path / version / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["sha256"]["feature_schema.json"] = hashlib.sha256(
            schema_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ActiveModelUnavailable) as error:
        load_active_model(tmp_path)

    assert error.value.error_code == error_code
    assert str(tmp_path) not in str(error.value)


def test_repeated_reads_are_consistent_and_reuse_loaded_model(tmp_path: Path) -> None:
    version = _make_package(tmp_path)
    activate_model(version, tmp_path)

    first = load_active_model(tmp_path)
    second = load_active_model(tmp_path)

    assert first.version == second.version
    assert first.model is second.model
    assert first.summary == second.summary
