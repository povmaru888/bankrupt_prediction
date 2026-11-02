"""Tests for the canonical ordered feature schema."""

from __future__ import annotations

import json
from pathlib import Path

from bankruptcy_prediction.data.feature_schema import (
    SCHEMA_VERSION,
    FeatureSchema,
    build_feature_schema,
)
from bankruptcy_prediction.data.training_loader import load_training_data

ROOT = Path(__file__).resolve().parents[2]


def test_schema_normalizes_names_preserves_order_and_serializes(
    feature_names: tuple[str, ...],
) -> None:
    dataset = load_training_data(ROOT / "Database" / "bankrupt_data.csv")
    schema = build_feature_schema(dataset)

    assert schema.schema_version == SCHEMA_VERSION == "1.0"
    assert schema.target_column == "Bankrupt?"
    assert tuple(field.name for field in schema.features) == feature_names
    assert len(schema.features) == 95
    assert len({field.name for field in schema.features}) == 95
    assert all(field.data_type == "number" for field in schema.features)
    assert {field.source_dtype for field in schema.features} == {"float64", "int64"}
    assert all(field.nullable is True for field in schema.features)

    restored = FeatureSchema.from_dict(json.loads(schema.to_json()))
    assert restored == schema
    assert json.loads(schema.to_json())["schema_version"] == "1.0"


def test_schema_keeps_case_and_internal_whitespace() -> None:
    from dataclasses import replace

    dataset = load_training_data(ROOT / "Database" / "bankrupt_data.csv")
    source = dataset.features.copy()
    source.columns = ["  Keep  Internal Case  ", *source.columns[1:]]
    schema = build_feature_schema(replace(dataset, features=source))

    assert schema.features[0].name == "Keep  Internal Case"
