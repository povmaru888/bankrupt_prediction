"""Tests for partial-success row quality validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bankruptcy_prediction.data.row_quality import validate_rows


def test_allows_up_to_nine_missing_values(
    valid_batch: pd.DataFrame, nine_missing_batch: pd.DataFrame
) -> None:
    clean = validate_rows(valid_batch)
    with_missing = validate_rows(nine_missing_batch)

    assert clean.valid_mask.tolist() == [True, True]
    assert with_missing.valid_mask.tolist() == [True, True]
    assert with_missing.missing_count.tolist() == [9, 0]
    assert with_missing.error_code.isna().all()


def test_ten_missing_values_fail_only_that_row(
    ten_missing_batch: pd.DataFrame,
) -> None:
    result = validate_rows(ten_missing_batch)

    assert result.valid_mask.tolist() == [False, True]
    assert result.error_code.tolist() == ["TOO_MANY_MISSING_VALUES", pd.NA]
    assert result.missing_count.tolist() == [10, 0]
    assert result.frame["record_id"].tolist() == ["fixture-001", "fixture-002"]


def test_non_numeric_and_positive_negative_infinity_are_row_errors(
    valid_batch: pd.DataFrame,
) -> None:
    batch = valid_batch.copy()
    batch[batch.columns[1]] = batch[batch.columns[1]].astype(object)
    batch.loc[0, batch.columns[1]] = "not-a-number"
    batch.loc[1, batch.columns[2]] = np.inf

    result = validate_rows(batch)

    assert result.valid_mask.tolist() == [False, False]
    assert result.error_code.tolist() == ["NON_NUMERIC_VALUE", "NON_FINITE_VALUE"]

    negative = valid_batch.copy()
    negative.loc[0, negative.columns[1]] = -np.inf
    assert validate_rows(negative).error_code.iloc[0] == "NON_FINITE_VALUE"


def test_preserves_row_order_and_ids_for_partial_success(
    valid_batch: pd.DataFrame, ten_missing_batch: pd.DataFrame
) -> None:
    mixed = pd.concat([valid_batch.iloc[[1]], ten_missing_batch.iloc[[0]]], ignore_index=True)

    result = validate_rows(mixed)

    assert result.frame["record_id"].tolist() == ["fixture-002", "fixture-001"]
    assert result.valid_mask.tolist() == [True, False]
    assert result.row_status.tolist() == ["success", "failed"]
