"""Validation tests for the immutable settings loader."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from bankruptcy_prediction.settings import SettingsError, load_settings

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "config" / "training.yaml"
SERVICE = ROOT / "config" / "service.yaml"


def _write_yaml(path: Path, content: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _config(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_loads_baseline_into_immutable_typed_settings() -> None:
    settings = load_settings(TRAINING, SERVICE)

    assert settings.training.random_seed == 42
    assert settings.training.search.n_iter == 30
    assert settings.service.host == "127.0.0.1"
    with pytest.raises(FrozenInstanceError):
        settings.service.host = "0.0.0.0"  # type: ignore[misc]


def test_missing_required_value_has_clear_error(tmp_path: Path) -> None:
    training = _config(TRAINING)
    del training["random_seed"]

    with pytest.raises(SettingsError, match="Missing required setting: training.random_seed"):
        load_settings(_write_yaml(tmp_path / "training.yaml", training), SERVICE)


def test_split_proportions_must_sum_to_one(tmp_path: Path) -> None:
    training = _config(TRAINING)
    training["split"]["test"] = 0.25  # type: ignore[index]

    with pytest.raises(SettingsError, match="proportions must sum to 1"):
        load_settings(_write_yaml(tmp_path / "training.yaml", training), SERVICE)


@pytest.mark.parametrize("port", [0, 65536, True, "8000"])
def test_service_port_must_be_valid_integer(tmp_path: Path, port: object) -> None:
    service = _config(SERVICE)
    service["api_port"] = port

    with pytest.raises(SettingsError, match="service.api_port"):
        load_settings(TRAINING, _write_yaml(tmp_path / "service.yaml", service))


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "https://example.test"])
def test_service_host_must_be_loopback(tmp_path: Path, host: str) -> None:
    service = _config(SERVICE)
    service["host"] = host

    with pytest.raises(SettingsError, match="service.host must be '127.0.0.1'"):
        load_settings(TRAINING, _write_yaml(tmp_path / "service.yaml", service))


@pytest.mark.parametrize("field", ["max_upload_size_mb", "max_rows"])
def test_service_limits_must_be_positive(tmp_path: Path, field: str) -> None:
    service = _config(SERVICE)
    service[field] = -1

    with pytest.raises(SettingsError, match=f"service.{field} must be an integer >= 1"):
        load_settings(TRAINING, _write_yaml(tmp_path / "service.yaml", service))
