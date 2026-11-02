"""Read-only loader and audit for the fixed MVP training dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

TARGET_COLUMN = "Bankrupt?"
EXPECTED_ROWS = 6819
EXPECTED_COLUMNS = 96
EXPECTED_POSITIVES = 220


class DatasetLoadError(ValueError):
    """Raised when the training source does not match the approved baseline."""


@dataclass(frozen=True)
class DatasetAudit:
    row_count: int
    column_count: int
    feature_count: int
    positive_count: int
    negative_count: int
    source_sha256: str


@dataclass(frozen=True)
class TrainingDataset:
    features: pd.DataFrame
    labels: pd.Series
    audit: DatasetAudit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(path: str | Path, *, test_only: bool = False) -> TrainingDataset:
    """Load and audit the SDD dataset; test-only mode permits a small schema-complete fixture."""
    if not isinstance(test_only, bool):
        raise ValueError("test_only must be a boolean")
    source = Path(path)
    try:
        frame = pd.read_csv(source, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetLoadError(f"Unable to read training CSV: {exc}") from exc

    if frame.shape[1] != EXPECTED_COLUMNS or (not test_only and frame.shape[0] != EXPECTED_ROWS):
        raise DatasetLoadError(
            f"Expected {'at least one' if test_only else EXPECTED_ROWS} rows and "
            f"{EXPECTED_COLUMNS} columns; "
            f"received {frame.shape[0]} rows and {frame.shape[1]} columns"
        )
    if frame.empty:
        raise DatasetLoadError("Training data must contain at least one row")
    if list(frame.columns).count(TARGET_COLUMN) != 1:
        raise DatasetLoadError(f"Expected exactly one target column named {TARGET_COLUMN!r}")

    labels = frame[TARGET_COLUMN]
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise DatasetLoadError(
            f"{TARGET_COLUMN!r} must contain only non-null binary values 0 and 1"
        )

    features = frame.drop(columns=[TARGET_COLUMN])
    non_numeric = [name for name, dtype in features.dtypes.items() if not is_numeric_dtype(dtype)]
    if non_numeric:
        raise DatasetLoadError(f"Non-numeric feature columns found: {non_numeric}")
    if features.isna().any().any():
        raise DatasetLoadError("Training features contain missing values")

    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not test_only and positives != EXPECTED_POSITIVES:
        raise DatasetLoadError(f"Expected {EXPECTED_POSITIVES} positive rows; received {positives}")

    return TrainingDataset(
        features=features,
        labels=labels,
        audit=DatasetAudit(
            row_count=len(frame),
            column_count=frame.shape[1],
            feature_count=features.shape[1],
            positive_count=positives,
            negative_count=negatives,
            source_sha256=_sha256(source),
        ),
    )
