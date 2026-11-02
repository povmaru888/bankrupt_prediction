"""Tests for generated and supplied per-batch record identifiers."""

from __future__ import annotations

import pandas as pd
import pytest

from bankruptcy_prediction.data.record_ids import RecordIdError, ensure_record_ids


def test_generates_six_digit_ids_when_column_is_missing() -> None:
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    identifiers = ensure_record_ids(frame)

    assert identifiers.tolist() == ["row_000001", "row_000002", "row_000003"]
    assert identifiers.index.equals(frame.index)
    assert "record_id" not in frame.columns


@pytest.mark.parametrize("values", [[None, "valid"], ["", "valid"], ["  ", "valid"]])
def test_rejects_missing_or_blank_identifiers(values: list[object]) -> None:
    frame = pd.DataFrame({"record_id": values})

    with pytest.raises(RecordIdError, match="must be non-empty"):
        ensure_record_ids(frame)


@pytest.mark.parametrize("values", [["same", "same"], [1, "1"]])
def test_rejects_duplicate_identifiers_after_string_conversion(values: list[object]) -> None:
    frame = pd.DataFrame({"record_id": values})

    with pytest.raises(RecordIdError, match="must be unique"):
        ensure_record_ids(frame)


def test_preserves_supplied_unique_identifiers_in_source_order() -> None:
    frame = pd.DataFrame({"record_id": ["third", "first", "second"]})

    assert ensure_record_ids(frame).tolist() == ["third", "first", "second"]


def test_generates_stable_ids_for_ten_thousand_rows() -> None:
    frame = pd.DataFrame({"feature": [0.0] * 10_000})

    identifiers = ensure_record_ids(frame)

    assert len(identifiers) == 10_000
    assert identifiers.iloc[0] == "row_000001"
    assert identifiers.iloc[-1] == "row_010000"
