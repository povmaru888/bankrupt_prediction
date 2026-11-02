"""Stable JSON and DataFrame projections for batch scoring results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from bankruptcy_prediction.contracts import BatchSummary, RowStatus, WarningCode

RESULT_COLUMNS = (
    "record_id",
    "model_version",
    "confidence_score",
    "batch_percentile",
    "threshold_used",
    "risk_class",
    "top_10_percent_flag",
    "row_status",
    "warning_codes",
    "positive_factors",
    "negative_factors",
    "error_code",
    "error_message",
)


class RiskClass(StrEnum):
    HIGH_RISK = "high_risk"
    NOT_HIGH_RISK = "not_high_risk"


@dataclass(frozen=True)
class FactorContribution:
    feature_name: str
    contribution: float

    def __post_init__(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("feature_name must be non-empty")
        if not pd.notna(self.contribution):
            raise ValueError("contribution must be finite")
        if self.contribution == float("inf") or self.contribution == float("-inf"):
            raise ValueError("contribution must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"feature_name": self.feature_name, "contribution": self.contribution}


@dataclass(frozen=True)
class ScoringResultRow:
    record_id: str
    model_version: str
    row_status: RowStatus
    threshold_used: float
    confidence_score: float | None = None
    batch_percentile: float | None = None
    risk_class: RiskClass | None = None
    top_10_percent_flag: bool | None = None
    warning_codes: tuple[WarningCode, ...] = ()
    positive_factors: tuple[FactorContribution, ...] | None = None
    negative_factors: tuple[FactorContribution, ...] | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not 0 <= self.threshold_used <= 1:
            raise ValueError("threshold_used must be between 0 and 1")
        if self.row_status is RowStatus.SUCCESS:
            if self.confidence_score is None or not 0 <= self.confidence_score <= 1:
                raise ValueError("successful rows require confidence_score between 0 and 1")
            if self.batch_percentile is None or not 0 <= self.batch_percentile <= 100:
                raise ValueError("successful rows require batch_percentile between 0 and 100")
            if self.risk_class is None or self.top_10_percent_flag is None:
                raise ValueError("successful rows require risk_class and top_10_percent_flag")
            if self.positive_factors is None or self.negative_factors is None:
                raise ValueError("successful rows require positive and negative factor lists")
            if len(self.positive_factors) > 3 or len(self.negative_factors) > 3:
                raise ValueError("each explanation list may contain at most three factors")
            if any(factor.contribution <= 0 for factor in self.positive_factors):
                raise ValueError("positive factors must have positive contributions")
            if any(factor.contribution >= 0 for factor in self.negative_factors):
                raise ValueError("negative factors must have negative contributions")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("successful rows cannot have error fields")
        else:
            if any(
                value is not None
                for value in (
                    self.confidence_score,
                    self.batch_percentile,
                    self.risk_class,
                    self.top_10_percent_flag,
                    self.positive_factors,
                    self.negative_factors,
                )
            ):
                raise ValueError("failed rows cannot contain scoring or explanation values")
            if not self.error_code or not self.error_message:
                raise ValueError("failed rows require error_code and error_message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "model_version": self.model_version,
            "confidence_score": self.confidence_score,
            "batch_percentile": self.batch_percentile,
            "threshold_used": self.threshold_used,
            "risk_class": self.risk_class.value if self.risk_class else None,
            "top_10_percent_flag": self.top_10_percent_flag,
            "row_status": self.row_status.value,
            "warning_codes": [warning.value for warning in self.warning_codes],
            "positive_factors": (
                [factor.to_dict() for factor in self.positive_factors]
                if self.positive_factors is not None
                else None
            ),
            "negative_factors": (
                [factor.to_dict() for factor in self.negative_factors]
                if self.negative_factors is not None
                else None
            ),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoringResultRow:
        return cls(
            record_id=payload["record_id"],
            model_version=payload["model_version"],
            row_status=RowStatus(payload["row_status"]),
            threshold_used=payload["threshold_used"],
            confidence_score=payload.get("confidence_score"),
            batch_percentile=payload.get("batch_percentile"),
            risk_class=RiskClass(payload["risk_class"]) if payload.get("risk_class") else None,
            top_10_percent_flag=payload.get("top_10_percent_flag"),
            warning_codes=tuple(WarningCode(code) for code in payload.get("warning_codes", [])),
            positive_factors=(
                tuple(FactorContribution(**factor) for factor in payload["positive_factors"])
                if payload.get("positive_factors") is not None
                else None
            ),
            negative_factors=(
                tuple(FactorContribution(**factor) for factor in payload["negative_factors"])
                if payload.get("negative_factors") is not None
                else None
            ),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
        )


@dataclass(frozen=True)
class ScoringResult:
    summary: BatchSummary
    rows: tuple[ScoringResultRow, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != self.summary.total_rows:
            raise ValueError("result row count must match batch summary")

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary.to_dict(), "rows": [row.to_dict() for row in self.rows]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for row in self.rows:
            record = row.to_dict()
            for key in ("warning_codes", "positive_factors", "negative_factors"):
                value = record[key]
                record[key] = json.dumps(value, ensure_ascii=False) if value is not None else None
            records.append(record)
        return pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoringResult:
        return cls(
            summary=BatchSummary.from_dict(payload["summary"]),
            rows=tuple(ScoringResultRow.from_dict(row) for row in payload["rows"]),
        )


def results_from_dataframe(frame: pd.DataFrame, summary: BatchSummary) -> ScoringResult:
    """Validate a fixed-order DataFrame projection and convert it to typed rows."""
    if tuple(frame.columns) != RESULT_COLUMNS:
        raise ValueError("result DataFrame columns do not match the fixed result schema")
    decoded_rows: list[ScoringResultRow] = []
    nullable_fields = (
        "confidence_score",
        "batch_percentile",
        "risk_class",
        "top_10_percent_flag",
        "positive_factors",
        "negative_factors",
        "error_code",
        "error_message",
    )
    json_fields = ("warning_codes", "positive_factors", "negative_factors")
    for raw_row in frame.to_dict(orient="records"):
        row = dict(raw_row)
        for field in nullable_fields:
            value = row.get(field)
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                row[field] = None
        for field in json_fields:
            if isinstance(row[field], str):
                row[field] = json.loads(row[field])
        decoded_rows.append(ScoringResultRow.from_dict(row))
    return ScoringResult(summary=summary, rows=tuple(decoded_rows))
