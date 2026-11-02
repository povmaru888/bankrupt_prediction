"""Small deterministic fixtures based on the official feature-name schema."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def feature_names() -> tuple[str, ...]:
    with (FIXTURE_DIR / "feature_schema.csv").open(encoding="utf-8-sig", newline="") as stream:
        headers = next(csv.reader(stream))
    return tuple(name.strip() for name in headers if name != "record_id")


@pytest.fixture
def valid_batch(feature_names: tuple[str, ...]) -> pd.DataFrame:
    values = [[0.25 for _ in feature_names], [0.5 for _ in feature_names]]
    return pd.DataFrame(values, columns=feature_names).assign(
        record_id=["fixture-001", "fixture-002"]
    )[["record_id", *feature_names]]


@pytest.fixture
def missing_feature_batch(valid_batch: pd.DataFrame) -> pd.DataFrame:
    return valid_batch.drop(columns=valid_batch.columns[-1])


@pytest.fixture
def unknown_feature_batch(valid_batch: pd.DataFrame) -> pd.DataFrame:
    return valid_batch.assign(unrecognized_feature=0.1)


@pytest.fixture
def nine_missing_batch(valid_batch: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    batch = valid_batch.copy()
    batch.loc[0, list(feature_names[:9])] = float("nan")
    return batch


@pytest.fixture
def ten_missing_batch(valid_batch: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    batch = valid_batch.copy()
    batch.loc[0, list(feature_names[:10])] = float("nan")
    return batch


@pytest.fixture
def infinite_value_batch(valid_batch: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    batch = valid_batch.copy()
    batch.loc[0, feature_names[0]] = float("inf")
    return batch


@pytest.fixture
def tied_score_rows() -> tuple[dict[str, object], ...]:
    return (
        {"record_id": "tie-first", "confidence_score": 0.75, "source_order": 0},
        {"record_id": "tie-second", "confidence_score": 0.75, "source_order": 1},
    )


@pytest.fixture
def formula_injection_batch(valid_batch: pd.DataFrame) -> pd.DataFrame:
    batch = valid_batch.copy()
    batch.loc[0, "record_id"] = '=HYPERLINK("https://example.invalid")'
    return batch
