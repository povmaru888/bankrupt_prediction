"""Tests for official training data loading, audit, and source immutability."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from bankruptcy_prediction.data.training_loader import DatasetLoadError, load_training_data

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CSV = ROOT / "Database" / "bankrupt_data.csv"
DAMAGED_CSV = ROOT / "tests" / "fixtures" / "damaged_training.csv"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_and_audits_official_dataset_without_modifying_it() -> None:
    before_hash = _file_sha256(OFFICIAL_CSV)
    dataset = load_training_data(OFFICIAL_CSV)
    after_hash = _file_sha256(OFFICIAL_CSV)

    assert dataset.features.shape == (6819, 95)
    assert dataset.labels.shape == (6819,)
    assert dataset.audit.row_count == 6819
    assert dataset.audit.column_count == 96
    assert dataset.audit.feature_count == 95
    assert dataset.audit.positive_count == 220
    assert dataset.audit.negative_count == 6599
    assert dataset.audit.source_sha256 == before_hash == after_hash
    assert all(str(dtype).startswith(("int", "float")) for dtype in dataset.features.dtypes)


def test_rejects_damaged_training_fixture() -> None:
    with pytest.raises(DatasetLoadError, match="Expected 6819 rows and 96 columns"):
        load_training_data(DAMAGED_CSV)


def test_test_only_mode_accepts_small_schema_complete_data(tmp_path: Path) -> None:
    columns = list(pd.read_csv(OFFICIAL_CSV, nrows=0).columns)
    frame = pd.DataFrame(0.0, index=range(20), columns=columns)
    frame["Bankrupt?"] = [0] * 16 + [1] * 4
    path = tmp_path / "test_only.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(DatasetLoadError, match="Expected 6819 rows"):
        load_training_data(path)
    dataset = load_training_data(path, test_only=True)

    assert dataset.features.shape == (20, 95)
    assert dataset.audit.positive_count == 4
    assert dataset.audit.row_count == 20
