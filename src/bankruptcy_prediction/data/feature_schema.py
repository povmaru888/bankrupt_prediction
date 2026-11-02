"""Canonical prediction feature schema derived from the audited training data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pandas.api.types import is_numeric_dtype

from bankruptcy_prediction.data.training_loader import TARGET_COLUMN, TrainingDataset

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureField:
    name: str
    data_type: str
    source_dtype: str
    nullable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "source_dtype": self.source_dtype,
            "nullable": self.nullable,
        }


@dataclass(frozen=True)
class FeatureSchema:
    schema_version: str
    target_column: str
    features: tuple[FeatureField, ...]

    def __post_init__(self) -> None:
        names = [feature.name for feature in self.features]
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        if not self.target_column.strip():
            raise ValueError("target_column must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique after normalization")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_column": self.target_column,
            "features": [feature.to_dict() for feature in self.features],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSchema:
        return cls(
            schema_version=payload["schema_version"],
            target_column=payload["target_column"],
            features=tuple(FeatureField(**item) for item in payload["features"]),
        )


def build_feature_schema(dataset: TrainingDataset) -> FeatureSchema:
    """Build ordered, whitespace-normalized input fields from the training dataset."""
    fields: list[FeatureField] = []
    for name, dtype in dataset.features.dtypes.items():
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("feature names cannot be empty after trimming whitespace")
        if not is_numeric_dtype(dtype):
            raise ValueError(f"feature {normalized_name!r} must be numeric")
        fields.append(
            FeatureField(
                name=normalized_name,
                data_type="number",
                source_dtype=str(dtype),
                nullable=True,
            )
        )
    return FeatureSchema(
        schema_version=SCHEMA_VERSION,
        target_column=TARGET_COLUMN,
        features=tuple(fields),
    )
