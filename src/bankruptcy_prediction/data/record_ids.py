"""Stable per-batch record identifier generation and validation."""

from __future__ import annotations

import pandas as pd

from bankruptcy_prediction.data.schema_validator import RECORD_ID_COLUMN


class RecordIdError(ValueError):
    """Raised when supplied record identifiers are empty or duplicated."""


def ensure_record_ids(frame: pd.DataFrame) -> pd.Series:
    """Return ordered string IDs, generating zero-padded IDs if absent."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if list(frame.columns).count(RECORD_ID_COLUMN) > 1:
        raise RecordIdError("record_id column must occur at most once")
    if RECORD_ID_COLUMN not in frame.columns:
        return pd.Series(
            (f"row_{position:06d}" for position in range(1, len(frame) + 1)),
            index=frame.index,
            name=RECORD_ID_COLUMN,
            dtype="string",
        )

    identifiers = frame[RECORD_ID_COLUMN].astype("string")
    empty = identifiers.isna() | identifiers.str.strip().eq("").fillna(False)
    if empty.any():
        raise RecordIdError("record_id values must be non-empty")
    if identifiers.duplicated().any():
        raise RecordIdError("record_id values must be unique within the batch")
    identifiers.name = RECORD_ID_COLUMN
    return identifiers
