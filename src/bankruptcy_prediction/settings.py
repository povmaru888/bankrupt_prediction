"""Typed, immutable loading and validation for local application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SettingsError(ValueError):
    """Raised when a settings file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class SplitSettings:
    strategy: str
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class SearchSettings:
    n_iter: int
    cv_folds: int
    scoring: str


@dataclass(frozen=True)
class CalibrationSettings:
    method: str
    cv_folds: int


@dataclass(frozen=True)
class TrainingSettings:
    data_path: str
    random_seed: int
    split: SplitSettings
    search: SearchSettings
    calibration: CalibrationSettings
    models: tuple[str, ...]
    use_smote: bool
    test_only: bool


@dataclass(frozen=True)
class ServiceSettings:
    host: str
    api_port: int
    ui_port: int
    max_upload_size_mb: int
    max_rows: int
    models_dir: str
    logs_dir: str
    temp_dir: str


@dataclass(frozen=True)
class AppSettings:
    training: TrainingSettings
    service: ServiceSettings


def _read_mapping(path: str | Path, kind: str) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
    except OSError as exc:
        raise SettingsError(f"Cannot read {kind} settings at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML in {kind} settings at {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise SettingsError(f"{kind} settings must be a YAML mapping")
    return content


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SettingsError(f"Missing required setting: {context}.{key}")
    return mapping[key]


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SettingsError(f"{context} must be a mapping")
    return value


def _integer(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SettingsError(f"{context} must be an integer >= {minimum}")
    return value


def _positive_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"{context} must be a non-empty string")
    return value


def _load_training(path: str | Path) -> TrainingSettings:
    raw = _read_mapping(path, "training")
    split_raw = _mapping(_required(raw, "split", "training"), "training.split")
    values: dict[str, float] = {}
    for name in ("train", "validation", "test"):
        value = _required(split_raw, name, "training.split")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"training.split.{name} must be a number between 0 and 1")
        if not 0 < value < 1:
            raise SettingsError(f"training.split.{name} must be greater than 0 and less than 1")
        values[name] = float(value)
    if abs(sum(values.values()) - 1.0) > 1e-9:
        raise SettingsError("training.split proportions must sum to 1")
    strategy = _required(split_raw, "strategy", "training.split")
    if strategy != "stratified":
        raise SettingsError("training.split.strategy must be 'stratified'")

    search_raw = _mapping(_required(raw, "search", "training"), "training.search")
    n_iter = _integer(
        _required(search_raw, "n_iter", "training.search"), "training.search.n_iter", 1
    )
    search_folds = _integer(
        _required(search_raw, "cv_folds", "training.search"),
        "training.search.cv_folds",
        2,
    )
    scoring = _required(search_raw, "scoring", "training.search")
    if scoring != "average_precision":
        raise SettingsError("training.search.scoring must be 'average_precision'")

    calibration_raw = _mapping(_required(raw, "calibration", "training"), "training.calibration")
    method = _required(calibration_raw, "method", "training.calibration")
    if method != "sigmoid":
        raise SettingsError("training.calibration.method must be 'sigmoid'")
    calibration_folds = _integer(
        _required(calibration_raw, "cv_folds", "training.calibration"),
        "training.calibration.cv_folds",
        2,
    )

    models_raw = _required(raw, "models", "training")
    if not isinstance(models_raw, list) or tuple(models_raw) != (
        "logistic_regression",
        "xgboost",
    ):
        raise SettingsError("training.models must contain logistic_regression and xgboost")
    use_smote = _required(raw, "use_smote", "training")
    if not isinstance(use_smote, bool) or use_smote:
        raise SettingsError("training.use_smote must be false")
    test_only = raw.get("test_only", False)
    if not isinstance(test_only, bool):
        raise SettingsError("training.test_only must be a boolean")

    return TrainingSettings(
        data_path=_positive_string(_required(raw, "data_path", "training"), "training.data_path"),
        random_seed=_integer(_required(raw, "random_seed", "training"), "training.random_seed"),
        split=SplitSettings("stratified", values["train"], values["validation"], values["test"]),
        search=SearchSettings(n_iter, search_folds, "average_precision"),
        calibration=CalibrationSettings("sigmoid", calibration_folds),
        models=tuple(models_raw),
        use_smote=False,
        test_only=test_only,
    )


def _load_service(path: str | Path) -> ServiceSettings:
    raw = _read_mapping(path, "service")
    host = _required(raw, "host", "service")
    if host != "127.0.0.1":
        raise SettingsError("service.host must be '127.0.0.1'")
    api_port = _integer(_required(raw, "api_port", "service"), "service.api_port", 1)
    ui_port = _integer(_required(raw, "ui_port", "service"), "service.ui_port", 1)
    if api_port > 65535 or ui_port > 65535:
        raise SettingsError("service.api_port and service.ui_port must be between 1 and 65535")
    if api_port == ui_port:
        raise SettingsError("service.api_port and service.ui_port must differ")
    return ServiceSettings(
        host=host,
        api_port=api_port,
        ui_port=ui_port,
        max_upload_size_mb=_integer(
            _required(raw, "max_upload_size_mb", "service"),
            "service.max_upload_size_mb",
            1,
        ),
        max_rows=_integer(_required(raw, "max_rows", "service"), "service.max_rows", 1),
        models_dir=_positive_string(_required(raw, "models_dir", "service"), "service.models_dir"),
        logs_dir=_positive_string(_required(raw, "logs_dir", "service"), "service.logs_dir"),
        temp_dir=_positive_string(_required(raw, "temp_dir", "service"), "service.temp_dir"),
    )


def load_settings(training_path: str | Path, service_path: str | Path) -> AppSettings:
    """Load and validate YAML files into an immutable application settings object."""
    return AppSettings(
        training=_load_training(training_path),
        service=_load_service(service_path),
    )
