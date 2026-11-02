"""Read-only active-model registry with checksum-gated, cached model loading."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib

from bankruptcy_prediction.model_package.verifier import (
    PackageVerificationError,
    VerifiedPackage,
    verify_package,
)

ACTIVE_SCHEMA_VERSION = 1
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ActiveModelUnavailable(RuntimeError):
    """Safe explanation for why no active model can currently serve predictions."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class ModelSummary:
    version: str
    model_family: str
    created_at: str
    test_only: bool
    default_threshold: float
    feature_count: int
    average_precision: float | None
    recall_at_top_10_percent: float | None
    mvp_passed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_family": self.model_family,
            "created_at": self.created_at,
            "test_only": self.test_only,
            "default_threshold": self.default_threshold,
            "feature_count": self.feature_count,
            "average_precision": self.average_precision,
            "recall_at_top_10_percent": self.recall_at_top_10_percent,
            "mvp_passed": self.mvp_passed,
        }


@dataclass(frozen=True)
class ActiveModel:
    version: str
    model: Any
    feature_names: tuple[str, ...]
    feature_schema: Mapping[str, Any]
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    default_threshold: float
    model_family: str
    summary: ModelSummary


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _read_active_version(models_root: Path) -> str:
    active_path = models_root / "active.json"
    if active_path.is_symlink():
        raise ActiveModelUnavailable("invalid_active_pointer", "active model pointer is invalid")
    try:
        payload = json.loads(active_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ActiveModelUnavailable("no_active_model", "no active model is configured") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActiveModelUnavailable(
            "invalid_active_pointer", "active model pointer is invalid"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != ACTIVE_SCHEMA_VERSION
        or not isinstance(payload.get("version"), str)
        or not payload["version"].strip()
        or not VERSION_PATTERN.fullmatch(payload["version"])
    ):
        raise ActiveModelUnavailable("invalid_active_pointer", "active model pointer is invalid")
    return payload["version"]


@lru_cache(maxsize=8)
def _load_verified_joblib(model_path: str, checksum: str) -> Any:
    """Only called after verify_package has checked this file against checksum."""
    del checksum  # Included in the cache key so changed bytes cannot reuse stale objects.
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise ActiveModelUnavailable(
            "model_load_failed", "verified model artifact could not be loaded"
        ) from exc


def _summary(package: VerifiedPackage) -> ModelSummary:
    manifest = package.manifest
    metrics = package.metrics
    threshold = float(manifest["threshold"]["default_threshold"])
    return ModelSummary(
        version=package.version,
        model_family=manifest["model_family"],
        created_at=str(manifest["created_at"]),
        test_only=manifest["test_only"],
        default_threshold=threshold,
        feature_count=len(package.feature_schema["feature_names"]),
        average_precision=(
            float(metrics["average_precision"])
            if isinstance(metrics.get("average_precision"), (int, float))
            else None
        ),
        recall_at_top_10_percent=(
            float(metrics["recall_at_top_10_percent"])
            if isinstance(metrics.get("recall_at_top_10_percent"), (int, float))
            else None
        ),
        mvp_passed=metrics.get("mvp_passed")
        if isinstance(metrics.get("mvp_passed"), bool)
        else None,
    )


def load_active_model(models_root: str | Path) -> ActiveModel:
    """Read active pointer, verify every artifact, then load and cache its trusted model."""
    root = Path(models_root)
    try:
        root_resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ActiveModelUnavailable("no_active_model", "models directory is unavailable") from exc
    version = _read_active_version(root_resolved)
    package_path = root_resolved / version
    try:
        package_path.resolve(strict=True).relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ActiveModelUnavailable(
            "invalid_active_model", "active model package is missing or failed verification"
        ) from exc
    try:
        package = verify_package(package_path)
    except PackageVerificationError as exc:
        raise ActiveModelUnavailable(
            "invalid_active_model", "active model package is missing or failed verification"
        ) from exc

    checksums = package.manifest["files"]["sha256"]
    model = _load_verified_joblib(str(package.model_path), checksums["model.joblib"])
    schema = _freeze(package.feature_schema)
    metrics = _freeze(package.metrics)
    manifest = _freeze(package.manifest)
    summary = _summary(package)
    return ActiveModel(
        version=package.version,
        model=model,
        feature_names=tuple(package.feature_schema["feature_names"]),
        feature_schema=schema,
        metrics=metrics,
        manifest=manifest,
        default_threshold=summary.default_threshold,
        model_family=summary.model_family,
        summary=summary,
    )
