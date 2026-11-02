"""Build an immutable, checksummed candidate model package atomically."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from bankruptcy_prediction.reporting.model_report import ModelReportContext, build_model_report

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SPLITS = ("train", "validation", "test")
CHART_FILES = (
    "precision_recall.png",
    "calibration.png",
    "confusion_matrix.png",
    "feature_importance.png",
)


@dataclass(frozen=True)
class CandidatePackageContext:
    model: Any
    model_family: str
    data_sha256: str
    split_seed: int
    split_indices: Mapping[str, Any]
    feature_names: tuple[str, ...]
    removed_columns: tuple[str, ...]
    package_versions: Mapping[str, str]
    model_parameters: Mapping[str, Any]
    calibration: Mapping[str, Any]
    threshold_result: Any
    metrics_report: Any
    report_context: ModelReportContext
    test_only: bool = False


@dataclass(frozen=True)
class ModelPackage:
    version: str
    path: Path
    manifest_path: Path
    metrics_path: Path
    report_path: Path


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_digest(indices: Any) -> tuple[int, str]:
    array = np.asarray(indices)
    if array.ndim != 1 or array.size == 0 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("each split index group must be a non-empty one-dimensional integer list")
    normalized = np.sort(array.astype(np.int64))
    if np.unique(normalized).size != normalized.size:
        raise ValueError("split indices must not repeat within a partition")
    encoded = json.dumps(normalized.tolist(), separators=(",", ":")).encode("ascii")
    return int(normalized.size), hashlib.sha256(encoded).hexdigest()


def _validate_context(context: CandidatePackageContext) -> tuple[float, dict[str, Any]]:
    if not isinstance(context, CandidatePackageContext):
        raise TypeError("context must be a CandidatePackageContext")
    if not SHA256_PATTERN.fullmatch(context.data_sha256):
        raise ValueError("data_sha256 must be a lowercase 64-character SHA-256 hex digest")
    if isinstance(context.split_seed, bool) or not isinstance(context.split_seed, int):
        raise ValueError("split_seed must be an integer")
    if set(context.split_indices) != set(REQUIRED_SPLITS):
        raise ValueError("split_indices must contain train, validation, and test partitions")
    summaries = {name: _index_digest(context.split_indices[name]) for name in REQUIRED_SPLITS}
    seen: set[int] = set()
    for split_name in REQUIRED_SPLITS:
        indices = set(np.asarray(context.split_indices[split_name], dtype=np.int64).tolist())
        if seen.intersection(indices):
            raise ValueError("train, validation, and test indices must be disjoint")
        seen.update(indices)
    names = context.feature_names
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and ordered")
    if not context.model_family.strip():
        raise ValueError("model_family must be non-empty")
    if not isinstance(context.test_only, bool):
        raise ValueError("test_only must be a boolean")
    threshold_payload = _jsonable(context.threshold_result)
    try:
        threshold = float(threshold_payload["default_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("threshold_result must expose default_threshold") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("default_threshold must be finite and between 0 and 1")
    if not isinstance(context.report_context, ModelReportContext):
        raise TypeError("report_context must be a ModelReportContext")
    if context.report_context.model_family != context.model_family:
        raise ValueError("report and package model_family must match")
    metrics_payload = _jsonable(context.metrics_report)
    report_metrics = _jsonable(context.report_context.metrics_report)
    if not isinstance(metrics_payload, Mapping) or not isinstance(report_metrics, Mapping):
        raise ValueError("metrics_report must be a mapping or expose to_dict()")
    if metrics_payload != report_metrics:
        raise ValueError("report and package metrics_report must match")
    if metrics_payload.get("threshold") != threshold:
        raise ValueError("metrics threshold must match the selected default threshold")
    return threshold, {
        split: {"row_count": summaries[split][0], "indices_sha256": summaries[split][1]}
        for split in REQUIRED_SPLITS
    }


def _version_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"candidate-{timestamp}-{uuid.uuid4().hex[:8]}"


def build_candidate_package(
    context: CandidatePackageContext, models_root: str | Path
) -> ModelPackage:
    """Create and atomically publish a candidate package; never changes active.json."""
    threshold, split_summary = _validate_context(context)
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    version = _version_id()
    destination = root / version
    if destination.exists():
        raise FileExistsError(f"model package version already exists: {version}")

    staging = Path(tempfile.mkdtemp(prefix=".candidate-staging-", dir=root))
    try:
        model_path = staging / "model.joblib"
        joblib.dump(context.model, model_path)
        schema_path = staging / "feature_schema.json"
        _write_json(
            schema_path,
            {
                "schema_version": 1,
                "feature_count": len(context.feature_names),
                "feature_names": list(context.feature_names),
                "removed_columns": list(context.removed_columns),
            },
        )
        metrics_path = staging / "metrics.json"
        _write_json(metrics_path, _jsonable(context.metrics_report))

        report_path = build_model_report(context.report_context, staging)
        chart_paths = [staging / "charts" / name for name in CHART_FILES]
        required_files = [model_path, schema_path, metrics_path, report_path, *chart_paths]
        if any(not path.is_file() for path in required_files):
            raise RuntimeError("candidate model package is missing a required artifact")

        relative_checksums = {
            path.relative_to(staging).as_posix(): _file_sha256(path) for path in required_files
        }
        manifest = {
            "manifest_version": 1,
            "version": version,
            "status": "candidate",
            "test_only": context.test_only,
            "created_at": datetime.now(UTC).isoformat(),
            "model_family": context.model_family,
            "data_sha256": context.data_sha256,
            "split": {"seed": context.split_seed, "partitions": split_summary},
            "feature_names": list(context.feature_names),
            "removed_columns": list(context.removed_columns),
            "package_versions": context.package_versions,
            "model_parameters": context.model_parameters,
            "calibration": context.calibration,
            "threshold": {"default_threshold": threshold, "selection": context.threshold_result},
            "metrics": context.metrics_report,
            "files": {"sha256": relative_checksums},
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        if destination.exists():
            raise FileExistsError(f"model package version already exists: {version}")
        os.replace(staging, destination)
    except Exception:
        # The staging path is freshly created under models_root and is the only cleanup target.
        if staging.exists() and staging.parent == root:
            shutil.rmtree(staging)
        raise

    return ModelPackage(
        version=version,
        path=destination,
        manifest_path=destination / "manifest.json",
        metrics_path=destination / "metrics.json",
        report_path=destination / "model_report.html",
    )
