"""Format immutable scoring results for JSON and CSV API responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from bankruptcy_prediction.export.csv_exporter import export_results_csv
from bankruptcy_prediction.scoring.result_models import RiskClass, ScoringResult


class ResponseFormatError(ValueError):
    """A file-level response formatting error with an API-safe envelope."""

    def __init__(self, error_code: str, message: str, remediation: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details: dict[str, Any] = {}
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class JsonResponse:
    body: dict[str, Any]
    media_type: str = "application/json"
    status_code: int = 200

    def to_json(self) -> str:
        return json.dumps(self.body, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class CsvResponse:
    body: bytes
    media_type: str = "text/csv; charset=utf-8"
    filename: str = "scoring_results.csv"
    status_code: int = 200


def _validate_threshold(threshold: float | None, fallback: float) -> float:
    value = fallback if threshold is None else threshold
    if isinstance(value, bool):
        raise ResponseFormatError(
            "INVALID_THRESHOLD", "threshold 必須是 0 至 1 的數字。", "請提供介於 0 與 1 的門檻。"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResponseFormatError(
            "INVALID_THRESHOLD", "threshold 必須是 0 至 1 的數字。", "請提供介於 0 與 1 的門檻。"
        ) from exc
    if not np.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ResponseFormatError(
            "INVALID_THRESHOLD", "threshold 必須是 0 至 1 的數字。", "請提供介於 0 與 1 的門檻。"
        )
    return numeric


def _apply_threshold(result: ScoringResult, threshold: float) -> ScoringResult:
    rows = tuple(
        replace(
            row,
            threshold_used=threshold,
            risk_class=(
                None
                if row.confidence_score is None
                else (
                    RiskClass.HIGH_RISK
                    if row.confidence_score >= threshold
                    else RiskClass.NOT_HIGH_RISK
                )
            ),
        )
        for row in result.rows
    )
    return ScoringResult(summary=replace(result.summary, threshold_used=threshold), rows=rows)


def format_score_response(
    result: ScoringResult,
    format: Literal["json", "csv"] | str,
    threshold: float | None = None,
) -> JsonResponse | CsvResponse:
    """Apply an ephemeral threshold and format a result without mutating it."""
    if not isinstance(result, ScoringResult):
        raise TypeError("result must be a ScoringResult")
    response_format = str(format).lower()
    if response_format not in {"json", "csv"}:
        raise ResponseFormatError(
            "UNSUPPORTED_FORMAT", "不支援的回應格式。", "請使用 json 或 csv。"
        )
    used_threshold = _validate_threshold(threshold, result.summary.threshold_used)
    adjusted = _apply_threshold(result, used_threshold)
    if response_format == "json":
        return JsonResponse(body=adjusted.to_dict())
    return CsvResponse(body=export_results_csv(adjusted))
