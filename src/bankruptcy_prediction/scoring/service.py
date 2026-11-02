"""Application service for synchronous, read-only batch scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from bankruptcy_prediction.contracts import BatchSummary, RowStatus, WarningCode
from bankruptcy_prediction.data.feature_schema import FeatureField, FeatureSchema
from bankruptcy_prediction.data.row_quality import validate_rows
from bankruptcy_prediction.data.schema_validator import RECORD_ID_COLUMN, validate_file_schema
from bankruptcy_prediction.explanation.shap_service import ExplanationBatch, explain
from bankruptcy_prediction.model_registry.registry import ActiveModel
from bankruptcy_prediction.scoring.result_models import (
    FactorContribution,
    RiskClass,
    ScoringResult,
    ScoringResultRow,
)

ScoreBatchResult = ScoringResult

_ERROR_MESSAGES = {
    "NON_NUMERIC_VALUE": "列含有非數字特徵值。",
    "NON_FINITE_VALUE": "列含有無限值。",
    "TOO_MANY_MISSING_VALUES": "列缺少 10 個以上特徵。",
}


@dataclass(frozen=True)
class _RowAnnotations:
    warnings: tuple[WarningCode, ...]


def _as_feature_schema(active_model: ActiveModel) -> FeatureSchema:
    """Adapt the model-package schema projection to the file validator contract."""
    schema = active_model.feature_schema
    if isinstance(schema, FeatureSchema):
        return schema
    if not isinstance(schema, Mapping):
        raise TypeError("active model feature_schema must be a mapping")
    names = schema.get("feature_names")
    if not isinstance(names, (tuple, list)) or not names:
        fields = schema.get("features")
        if isinstance(fields, (tuple, list)):
            names = [item["name"] for item in fields]
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("active model feature schema is invalid")
    return FeatureSchema(
        schema_version=str(schema.get("schema_version", "1.0")),
        target_column="Bankrupt?",
        features=tuple(
            FeatureField(
                name=name,
                data_type="number",
                source_dtype="float64",
                nullable=True,
            )
            for name in names
        ),
    )


def _range_warnings(
    active_model: ActiveModel, features: pd.DataFrame
) -> tuple[tuple[WarningCode, ...], ...]:
    """Use a fitted range monitor when the model package exposes one.

    Older package artifacts do not carry range statistics; in that case no
    range warning is fabricated and the fitted prediction pipeline remains the
    sole source of model transformations.
    """
    monitor = getattr(active_model, "range_monitor", None)
    if monitor is None:
        return tuple(() for _ in range(len(features)))
    matrix = monitor.flag_out_of_range(features)
    return tuple((WarningCode.OUT_OF_RANGE,) if codes else () for codes in matrix.warning_codes)


def _factor_list(factors: Any) -> tuple[FactorContribution, ...]:
    return tuple(
        FactorContribution(
            feature_name=factor.feature_name, contribution=float(factor.contribution)
        )
        for factor in factors
    )


def _percentiles(scores: np.ndarray) -> np.ndarray:
    """Return average-rank percentiles inside the valid scoring batch."""
    return (
        pd.Series(scores, dtype="float64")
        .rank(method="average", ascending=True, pct=True)
        .to_numpy()
        * 100
    )


def score_batch(
    active_model: ActiveModel,
    frame: pd.DataFrame,
    threshold: float | None = None,
) -> ScoreBatchResult:
    """Validate and score a batch using an already-fitted active model only."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    used_threshold = active_model.default_threshold if threshold is None else float(threshold)
    if not np.isfinite(used_threshold) or not 0 <= used_threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")

    validated = validate_file_schema(frame, _as_feature_schema(active_model))
    quality = validate_rows(validated.frame)
    records = quality.frame[RECORD_ID_COLUMN].astype(str).tolist()
    total = len(records)
    valid_positions = np.flatnonzero(quality.valid_mask.to_numpy(dtype=bool))
    valid_features = quality.frame.loc[quality.valid_mask].drop(columns=[RECORD_ID_COLUMN])

    scores = np.asarray([], dtype=float)
    explanations: ExplanationBatch | None = None
    percentiles = np.asarray([], dtype=float)
    top_positions: set[int] = set()
    valid_warnings: tuple[tuple[WarningCode, ...], ...] = ()
    if len(valid_positions):
        # The fitted pipeline owns imputation and all other transformations.
        probabilities = np.asarray(active_model.model.predict_proba(valid_features), dtype=float)
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != len(valid_positions)
            or probabilities.shape[1] < 2
        ):
            raise ValueError("active model returned an invalid probability matrix")
        scores = probabilities[:, 1]
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("active model returned invalid confidence scores")
        percentiles = _percentiles(scores)
        top_count = ceil(len(scores) * 0.1)
        ordered = np.argsort(-scores, kind="stable")[:top_count]
        top_positions = set(int(position) for position in ordered)
        explanations = explain(
            active_model.model_family, active_model.model, valid_features
        ).project(3)
        if len(explanations.rows) != len(valid_positions):
            raise ValueError("active model explanation count does not match score count")
        valid_warnings = tuple(
            tuple(
                dict.fromkeys(
                    (
                        [WarningCode.IMPUTED_VALUES]
                        if int(quality.missing_count.iloc[position])
                        else []
                    )
                    + list(range_codes)
                )
            )
            for position, range_codes in zip(
                valid_positions, _range_warnings(active_model, valid_features)
            )
        )

    result_rows: list[ScoringResultRow] = []
    valid_index = {int(position): offset for offset, position in enumerate(valid_positions)}
    for position, record_id in enumerate(records):
        if position not in valid_index:
            error_code = quality.error_code.iloc[position]
            code = str(error_code)
            result_rows.append(
                ScoringResultRow(
                    record_id=record_id,
                    model_version=active_model.version,
                    row_status=RowStatus.FAILED,
                    threshold_used=used_threshold,
                    error_code=code,
                    error_message=_ERROR_MESSAGES.get(code, "列未通過資料品質驗證。"),
                )
            )
            continue
        offset = valid_index[position]
        explanation = explanations.rows[offset]  # type: ignore[union-attr]
        score = float(scores[offset])
        result_rows.append(
            ScoringResultRow(
                record_id=record_id,
                model_version=active_model.version,
                row_status=RowStatus.SUCCESS,
                threshold_used=used_threshold,
                confidence_score=score,
                batch_percentile=float(percentiles[offset]),
                risk_class=(
                    RiskClass.HIGH_RISK if score >= used_threshold else RiskClass.NOT_HIGH_RISK
                ),
                top_10_percent_flag=offset in top_positions,
                warning_codes=valid_warnings[offset],
                positive_factors=_factor_list(explanation.positive_factors),
                negative_factors=_factor_list(explanation.negative_factors),
            )
        )
    succeeded = len(valid_positions)
    return ScoringResult(
        summary=BatchSummary(
            total_rows=total,
            succeeded_rows=succeeded,
            failed_rows=total - succeeded,
            model_version=active_model.version,
            threshold_used=used_threshold,
        ),
        rows=tuple(result_rows),
    )
