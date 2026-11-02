"""Row-level numeric and missing-value validation for prediction batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bankruptcy_prediction.contracts import RowStatus
from bankruptcy_prediction.data.record_ids import ensure_record_ids
from bankruptcy_prediction.data.schema_validator import RECORD_ID_COLUMN

MAX_MISSING_FEATURES = 9


@dataclass(frozen=True)
class RowValidationResult:
    frame: pd.DataFrame
    valid_mask: pd.Series
    row_status: pd.Series
    error_code: pd.Series
    missing_count: pd.Series


def validate_rows(batch: pd.DataFrame) -> RowValidationResult:
    """Coerce features to numeric and identify invalid rows without dropping any."""
    if not isinstance(batch, pd.DataFrame):
        raise TypeError("batch must be a pandas DataFrame")
    if batch.empty:
        raise ValueError("batch must contain at least one row")

    identifiers = ensure_record_ids(batch)
    if RECORD_ID_COLUMN in batch.columns:
        raw_features = batch.drop(columns=[RECORD_ID_COLUMN])
    else:
        raw_features = batch
    if raw_features.shape[1] == 0:
        raise ValueError("batch must contain at least one feature column")

    numeric = raw_features.apply(pd.to_numeric, errors="coerce")
    original_missing = raw_features.isna()
    non_numeric = numeric.isna() & ~original_missing
    missing_count = original_missing.sum(axis=1).astype("int64")
    non_finite = pd.DataFrame(
        np.isinf(numeric.to_numpy(dtype=float)), index=numeric.index, columns=numeric.columns
    )

    statuses: list[str] = []
    error_codes: list[str | None] = []
    for position in range(len(batch)):
        if non_numeric.iloc[position].any():
            statuses.append(RowStatus.FAILED.value)
            error_codes.append("NON_NUMERIC_VALUE")
        elif non_finite.iloc[position].any():
            statuses.append(RowStatus.FAILED.value)
            error_codes.append("NON_FINITE_VALUE")
        elif missing_count.iloc[position] > MAX_MISSING_FEATURES:
            statuses.append(RowStatus.FAILED.value)
            error_codes.append("TOO_MANY_MISSING_VALUES")
        else:
            statuses.append(RowStatus.SUCCESS.value)
            error_codes.append(None)

    row_status = pd.Series(statuses, index=batch.index, name="row_status", dtype="string")
    error_code = pd.Series(error_codes, index=batch.index, name="error_code", dtype="string")
    valid_mask = row_status.eq(RowStatus.SUCCESS.value).rename("valid")
    numeric_frame = numeric.copy()
    numeric_frame.insert(0, RECORD_ID_COLUMN, identifiers)
    return RowValidationResult(
        frame=numeric_frame,
        valid_mask=valid_mask,
        row_status=row_status,
        error_code=error_code,
        missing_count=missing_count.rename("missing_count"),
    )
