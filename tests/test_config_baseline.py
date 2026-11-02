"""Checks for the initial training and service configuration baseline."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_config(relative_path: str) -> dict[str, object]:
    with (ROOT / relative_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert isinstance(config, dict)
    return config


def test_training_configuration_matches_sdd_baseline() -> None:
    config = _load_config("config/training.yaml")
    split = config["split"]
    search = config["search"]
    calibration = config["calibration"]

    assert config["random_seed"] == 42
    assert isinstance(split, dict)
    assert split == {
        "strategy": "stratified",
        "train": 0.64,
        "validation": 0.16,
        "test": 0.2,
    }
    assert sum(split[key] for key in ("train", "validation", "test")) == 1.0
    assert isinstance(search, dict)
    assert search == {
        "n_iter": 30,
        "cv_folds": 5,
        "scoring": "average_precision",
    }
    assert isinstance(calibration, dict)
    assert calibration == {"method": "sigmoid", "cv_folds": 5}
    assert config["models"] == ["logistic_regression", "xgboost"]
    assert config["use_smote"] is False
    assert config["test_only"] is False


def test_service_configuration_is_local_and_within_limits() -> None:
    config = _load_config("config/service.yaml")

    assert config["host"] == "127.0.0.1"
    assert isinstance(config["api_port"], int) and 1 <= config["api_port"] <= 65535
    assert isinstance(config["ui_port"], int) and 1 <= config["ui_port"] <= 65535
    assert config["api_port"] != config["ui_port"]
    assert config["max_upload_size_mb"] == 10
    assert config["max_rows"] == 10000
    assert config["models_dir"] == "models"
    assert config["logs_dir"] == "logs"
    assert isinstance(config["temp_dir"], str) and config["temp_dir"]
