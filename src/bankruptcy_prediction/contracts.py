"""Shared row and batch result contracts with JSON-safe serialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RowStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class WarningCode(StrEnum):
    IMPUTED_VALUES = "IMPUTED_VALUES"
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class RowResult:
    record_id: str
    row_status: RowStatus
    warning_codes: tuple[WarningCode, ...] = ()
    confidence_score: float | None = None
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if self.row_status is RowStatus.FAILED:
            if not self.error_code or not self.error_message:
                raise ValueError("failed rows require error_code and error_message")
            if self.confidence_score is not None:
                raise ValueError("failed rows cannot have a confidence_score")
        elif self.error_code is not None or self.error_message is not None:
            raise ValueError("successful rows cannot have error fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "row_status": self.row_status.value,
            "warning_codes": [code.value for code in self.warning_codes],
            "confidence_score": self.confidence_score,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RowResult:
        return cls(
            record_id=payload["record_id"],
            row_status=RowStatus(payload["row_status"]),
            warning_codes=tuple(WarningCode(code) for code in payload["warning_codes"]),
            confidence_score=payload.get("confidence_score"),
            error_code=payload.get("error_code"),
            error_message=payload.get("error_message"),
        )


@dataclass(frozen=True)
class BatchSummary:
    total_rows: int
    succeeded_rows: int
    failed_rows: int
    model_version: str
    threshold_used: float

    def __post_init__(self) -> None:
        if min(self.total_rows, self.succeeded_rows, self.failed_rows) < 0:
            raise ValueError("row counts must be non-negative")
        if self.succeeded_rows + self.failed_rows != self.total_rows:
            raise ValueError("succeeded_rows + failed_rows must equal total_rows")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not 0 <= self.threshold_used <= 1:
            raise ValueError("threshold_used must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "succeeded_rows": self.succeeded_rows,
            "failed_rows": self.failed_rows,
            "model_version": self.model_version,
            "threshold_used": self.threshold_used,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BatchSummary:
        return cls(**payload)
