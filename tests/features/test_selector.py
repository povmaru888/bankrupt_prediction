"""Tests for deterministic feature cleanup and non-destructive audits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bankruptcy_prediction.features.selector import (
    APPROVED_DUPLICATES,
    CONSTANT_FEATURE,
    CORRELATION_THRESHOLD,
    FeatureSelector,
    selector_audit_to_dict,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def official_features() -> pd.DataFrame:
    return pd.read_csv(ROOT / "Database" / "bankrupt_data.csv").drop(columns="Bankrupt?")


def test_removes_only_the_three_approved_columns_and_keeps_order(
    official_features: pd.DataFrame,
) -> None:
    original_order = [name.strip() for name in official_features.columns]
    selector = FeatureSelector().fit(official_features)
    transformed = selector.transform(official_features)
    expected_removed = {
        CONSTANT_FEATURE,
        "Current Liability to Liability",
        "Current Liability to Equity",
    }

    assert selector.audit_.removed_columns == tuple(
        name for name in original_order if name in expected_removed
    )
    assert list(transformed.columns) == [
        name for name in original_order if name not in expected_removed
    ]
    assert transformed.shape == (6819, 92)
    assert selector.get_feature_names_out().tolist() == list(transformed.columns)


def test_correlation_threshold_reports_pairs_without_removing_fields(
    official_features: pd.DataFrame,
) -> None:
    selector = FeatureSelector().fit(official_features)
    audit = selector_audit_to_dict(selector)

    assert selector.audit_.correlation_threshold == 0.95
    assert CORRELATION_THRESHOLD == 0.95
    assert selector.audit_.high_correlation_pairs
    assert all(abs(pair.pearson_r) >= 0.95 for pair in selector.audit_.high_correlation_pairs)
    assert len(selector.audit_.retained_columns) == 92
    assert audit["correlation_threshold"] == 0.95
    assert len(audit["high_correlation_pairs"]) == len(selector.audit_.high_correlation_pairs)


def test_unexpected_exact_duplicates_are_audited_but_not_removed(
    official_features: pd.DataFrame,
) -> None:
    frame = official_features.copy()
    frame.columns = [name.strip() for name in frame.columns]
    frame["Current Ratio"] = frame["Quick Ratio"]
    selector = FeatureSelector().fit(frame)

    assert "Current Ratio" in selector.audit_.retained_columns
    assert any(
        pair.first_feature == "Current Ratio" and pair.duplicate_feature == "Quick Ratio"
        for pair in selector.audit_.unexpected_exact_duplicates
    )


def test_approved_duplicate_pair_names_are_fixed() -> None:
    assert APPROVED_DUPLICATES == {
        "Current Liability to Liability": "Current Liabilities/Liability",
        "Current Liability to Equity": "Current Liabilities/Equity",
    }


def test_rejects_changed_approved_duplicate_relation(official_features: pd.DataFrame) -> None:
    frame = official_features.copy()
    frame.columns = [name.strip() for name in frame.columns]
    frame["Current Liability to Liability"] = 0.0

    with pytest.raises(ValueError, match="no longer exactly matches"):
        FeatureSelector().fit(frame)
