"""File-level validation and canonical ordering for prediction CSV frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from bankruptcy_prediction.data.feature_schema import FeatureSchema
from bankruptcy_prediction.data.training_loader import TARGET_COLUMN
from bankruptcy_prediction.errors import ApiError

RECORD_ID_COLUMN = "record_id"


class FileSchemaError(ValueError):
    """Raised when a whole input file violates the feature-name contract."""

    def __init__(self, error: ApiError) -> None:
        super().__init__(error.message)
        self.api_error = error


@dataclass(frozen=True)
class ValidatedBatch:
    frame: pd.DataFrame
    schema_version: str


def _raise_schema_error(
    message: str,
    *,
    missing: list[str] | None = None,
    unknown: list[str] | None = None,
    duplicates: list[str] | None = None,
    target_column_present: bool = False,
) -> None:
    details: dict[str, Any] = {
        "missing_features": missing or [],
        "unknown_columns": unknown or [],
        "duplicate_columns": duplicates or [],
        "target_column_present": target_column_present,
    }
    raise FileSchemaError(
        ApiError(
            error_code="CSV_SCHEMA_INVALID",
            message=message,
            details=details,
            remediation="Correct the CSV headers to match the documented feature schema and retry.",
        )
    )


def validate_file_schema(frame: pd.DataFrame, schema: FeatureSchema) -> ValidatedBatch:
    """Validate file-level column names, then return a copied canonical-order batch."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        _raise_schema_error("CSV batch must contain at least one data row")
    if any(not isinstance(column, str) for column in frame.columns):
        _raise_schema_error("CSV column names must be strings")

    normalized = [column.strip() for column in frame.columns]
    duplicate_names = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicate_names:
        _raise_schema_error(
            "CSV contains duplicate column names after trimming whitespace",
            duplicates=duplicate_names,
        )

    feature_names = [field.name for field in schema.features]
    provided_names = set(normalized)
    expected_names = set(feature_names)
    missing = sorted(expected_names - provided_names)
    target_present = TARGET_COLUMN in provided_names
    allowed_extra = {RECORD_ID_COLUMN}
    unknown = sorted(provided_names - expected_names - allowed_extra)
    if missing or unknown or target_present:
        message = "CSV columns do not match the required feature schema"
        _raise_schema_error(
            message,
            missing=missing,
            unknown=unknown,
            target_column_present=target_present,
        )

    batch = frame.copy()
    batch.columns = normalized
    ordered_columns = (
        [RECORD_ID_COLUMN] if RECORD_ID_COLUMN in provided_names else []
    ) + feature_names
    return ValidatedBatch(frame=batch.loc[:, ordered_columns], schema_version=schema.schema_version)
