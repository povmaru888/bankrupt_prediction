"""Tests for file-level CSV schema rejection and canonical column ordering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.data.schema_validator import FileSchemaError, validate_file_schema
from bankruptcy_prediction.data.training_loader import load_training_data

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def schema():
    dataset = load_training_data(ROOT / "Database" / "bankrupt_data.csv")
    return build_feature_schema(dataset)


def test_accepts_whitespace_trim_and_returns_canonical_order(
    valid_batch: pd.DataFrame, schema
) -> None:
    shuffled = valid_batch.loc[:, list(reversed(valid_batch.columns))].copy()
    shuffled.columns = [f" {column} " for column in shuffled.columns]

    result = validate_file_schema(shuffled, schema)

    assert result.schema_version == schema.schema_version
    assert list(result.frame.columns) == ["record_id", *(field.name for field in schema.features)]
    assert result.frame.iloc[0]["record_id"] == "fixture-001"
    assert list(shuffled.columns) != list(result.frame.columns)


@pytest.mark.parametrize(
    "fixture_name,detail_key",
    [("missing_feature_batch", "missing_features"), ("unknown_feature_batch", "unknown_columns")],
)
def test_rejects_missing_or_unknown_columns_as_file_level_error(
    request: pytest.FixtureRequest, schema, fixture_name: str, detail_key: str
) -> None:
    frame = request.getfixturevalue(fixture_name)

    with pytest.raises(FileSchemaError) as error:
        validate_file_schema(frame, schema)

    assert error.value.api_error.error_code == "CSV_SCHEMA_INVALID"
    assert error.value.api_error.details[detail_key]
    assert set(error.value.api_error.to_dict()) == {
        "error_code",
        "message",
        "details",
        "remediation",
    }


def test_rejects_target_column_and_case_mismatch(valid_batch: pd.DataFrame, schema) -> None:
    with_target = valid_batch.assign(**{"Bankrupt?": 0})
    with pytest.raises(FileSchemaError) as error:
        validate_file_schema(with_target, schema)
    assert error.value.api_error.details["target_column_present"] is True

    case_mismatch = valid_batch.rename(
        columns={schema.features[0].name: schema.features[0].name.lower()}
    )
    with pytest.raises(FileSchemaError) as error:
        validate_file_schema(case_mismatch, schema)
    assert schema.features[0].name in error.value.api_error.details["missing_features"]


def test_rejects_empty_and_duplicate_normalized_headers(valid_batch: pd.DataFrame, schema) -> None:
    with pytest.raises(FileSchemaError, match="at least one data row"):
        validate_file_schema(valid_batch.iloc[0:0], schema)

    duplicate = valid_batch.copy()
    columns = list(duplicate.columns)
    columns[-1] = f" {columns[-2]} "
    duplicate.columns = columns
    with pytest.raises(FileSchemaError) as error:
        validate_file_schema(duplicate, schema)
    assert error.value.api_error.details["duplicate_columns"]
