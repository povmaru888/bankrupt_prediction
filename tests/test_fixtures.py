"""Checks fixture completeness and isolation from production data and models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_schema_has_95_official_feature_names(
    feature_names: tuple[str, ...],
) -> None:
    assert len(feature_names) == 95
    assert all(name == name.strip() for name in feature_names)
    assert "Net Income Flag" in feature_names
    assert "Current Liability to Liability" in feature_names
    assert "Current Liability to Equity" in feature_names


def test_batch_fixtures_are_deterministic_and_isolated(
    valid_batch: pd.DataFrame,
    missing_feature_batch: pd.DataFrame,
    unknown_feature_batch: pd.DataFrame,
    nine_missing_batch: pd.DataFrame,
    ten_missing_batch: pd.DataFrame,
    infinite_value_batch: pd.DataFrame,
    tied_score_rows: tuple[dict[str, object], ...],
    formula_injection_batch: pd.DataFrame,
) -> None:
    assert valid_batch.shape == (2, 96)
    assert missing_feature_batch.shape[1] == 95
    assert "unrecognized_feature" in unknown_feature_batch
    assert nine_missing_batch.iloc[0].isna().sum() == 9
    assert ten_missing_batch.iloc[0].isna().sum() == 10
    assert infinite_value_batch.iloc[0, 1] == float("inf")
    assert [row["record_id"] for row in tied_score_rows] == ["tie-first", "tie-second"]
    assert str(formula_injection_batch.loc[0, "record_id"]).startswith("=")
    assert (ROOT / "tests/fixtures/feature_schema.csv").is_file()
    assert not str(ROOT / "tests/fixtures").startswith(str(ROOT / "models"))
    assert not (ROOT / "models" / "test_fixture.joblib").exists()
