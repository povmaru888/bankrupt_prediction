"""Verify model-package schemas and checksums without loading serialized models."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FILES = frozenset(
    {
        "model.joblib",
        "feature_schema.json",
        "metrics.json",
        "model_report.html",
        "charts/precision_recall.png",
        "charts/calibration.png",
        "charts/confusion_matrix.png",
        "charts/feature_importance.png",
    }
)
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "version",
        "status",
        "test_only",
        "created_at",
        "model_family",
        "data_sha256",
        "split",
        "feature_names",
        "removed_columns",
        "package_versions",
        "model_parameters",
        "calibration",
        "threshold",
        "metrics",
        "files",
    }
)


class PackageVerificationError(ValueError):
    """Raised with a safe, path-free message when a model package is invalid."""


@dataclass(frozen=True)
class VerifiedPackage:
    path: Path
    version: str
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    metrics: dict[str, Any]
    model_path: Path


def _fail(reason: str) -> PackageVerificationError:
    return PackageVerificationError(f"model package verification failed: {reason}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"{label} is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise _fail(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _fail("a required package file cannot be read") from exc
    return digest.hexdigest()


def _safe_artifact(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in relative:
        raise _fail("manifest contains an unsafe artifact path")
    artifact = root / relative_path
    try:
        resolved = artifact.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise _fail("a required package file is missing or escapes the package") from exc
    if not resolved.is_file():
        raise _fail("a required package artifact is not a regular file")
    return resolved


def _validate_manifest(manifest: dict[str, Any], root: Path) -> dict[str, str]:
    if not REQUIRED_MANIFEST_FIELDS <= set(manifest):
        raise _fail("manifest is missing required fields")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise _fail("manifest schema version is unsupported")
    if manifest["version"] != root.name or not isinstance(manifest["version"], str):
        raise _fail("manifest version does not match its package directory")
    if manifest["status"] != "candidate":
        raise _fail("manifest status is not candidate")
    if not isinstance(manifest["test_only"], bool):
        raise _fail("manifest test_only flag is invalid")
    if not isinstance(manifest["model_family"], str) or not manifest["model_family"].strip():
        raise _fail("manifest model_family is invalid")
    if not isinstance(manifest["data_sha256"], str) or not SHA256_PATTERN.fullmatch(
        manifest["data_sha256"]
    ):
        raise _fail("manifest data hash is invalid")
    split = manifest["split"]
    if not isinstance(split, dict) or not isinstance(split.get("seed"), int):
        raise _fail("manifest split metadata is invalid")
    partitions = split.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise _fail("manifest split partitions are incomplete")
    for partition in partitions.values():
        if (
            not isinstance(partition, dict)
            or not isinstance(partition.get("row_count"), int)
            or partition["row_count"] < 1
            or not isinstance(partition.get("indices_sha256"), str)
            or not SHA256_PATTERN.fullmatch(partition["indices_sha256"])
        ):
            raise _fail("manifest split partition summary is invalid")
    names = manifest["feature_names"]
    removed = manifest["removed_columns"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise _fail("manifest feature order is invalid")
    if not isinstance(removed, list) or any(not isinstance(name, str) for name in removed):
        raise _fail("manifest removed_columns is invalid")
    if not isinstance(manifest["package_versions"], dict):
        raise _fail("manifest package_versions is invalid")
    if not isinstance(manifest["model_parameters"], dict):
        raise _fail("manifest model_parameters is invalid")
    if not isinstance(manifest["calibration"], dict):
        raise _fail("manifest calibration metadata is invalid")
    threshold = manifest["threshold"]
    if (
        not isinstance(threshold, dict)
        or isinstance(threshold.get("default_threshold"), bool)
        or not isinstance(threshold.get("default_threshold"), (float, int))
        or not 0 <= threshold["default_threshold"] <= 1
        or not isinstance(threshold.get("selection"), dict)
    ):
        raise _fail("manifest threshold metadata is invalid")
    if not isinstance(manifest["metrics"], dict):
        raise _fail("manifest metrics are invalid")
    files = manifest["files"]
    checksums = files.get("sha256") if isinstance(files, dict) else None
    if not isinstance(checksums, dict) or set(checksums) != REQUIRED_FILES:
        raise _fail("manifest file checksum list is incomplete or unexpected")
    if any(
        not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
        for value in checksums.values()
    ):
        raise _fail("manifest contains an invalid file checksum")
    return checksums


def _validate_feature_schema(schema: dict[str, Any], manifest: dict[str, Any]) -> None:
    if schema.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise _fail("feature schema version is unsupported")
    names = schema.get("feature_names")
    removed = schema.get("removed_columns")
    if (
        schema.get("feature_count") != len(manifest["feature_names"])
        or names != manifest["feature_names"]
        or removed != manifest["removed_columns"]
    ):
        raise _fail("feature schema does not match manifest")


def _validate_offline_report(path: Path) -> None:
    try:
        report = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail("model report cannot be read") from exc
    for chart in sorted(REQUIRED_FILES):
        if chart.startswith("charts/") and f'src="{chart}"' not in report:
            raise _fail("model report contains a missing or invalid chart reference")
    if "https://" in report or "http://" in report:
        raise _fail("model report must not reference external resources")


def verify_package(path: str | Path) -> VerifiedPackage:
    """Validate metadata, paths, files and checksums; never deserialize model.joblib."""
    try:
        root = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail("package directory is unavailable") from exc
    if not root.is_dir():
        raise _fail("package path is not a directory")

    manifest_path = _safe_artifact(root, "manifest.json")
    manifest = _read_json(manifest_path, "manifest")
    checksums = _validate_manifest(manifest, root)
    artifacts = {relative: _safe_artifact(root, relative) for relative in REQUIRED_FILES}
    for relative, artifact in artifacts.items():
        if _sha256(artifact) != checksums[relative]:
            raise _fail(f"package file checksum does not match: {relative}")

    schema = _read_json(artifacts["feature_schema.json"], "feature_schema.json")
    metrics = _read_json(artifacts["metrics.json"], "metrics.json")
    _validate_feature_schema(schema, manifest)
    if metrics != manifest["metrics"]:
        raise _fail("metrics file does not match manifest")
    _validate_offline_report(artifacts["model_report.html"])
    return VerifiedPackage(
        path=root,
        version=manifest["version"],
        manifest=manifest,
        feature_schema=schema,
        metrics=metrics,
        model_path=artifacts["model.joblib"],
    )
