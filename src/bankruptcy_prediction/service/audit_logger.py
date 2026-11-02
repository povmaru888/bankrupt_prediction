"""Scoring audit-log adapter that exposes no row-level data."""

from __future__ import annotations

from collections import Counter

from bankruptcy_prediction.logging_config import log_score_summary
from bankruptcy_prediction.scoring.result_models import ScoringResult


def record_score_request(result: ScoringResult, *, elapsed_ms: int, request_id: str) -> str:
    """Persist an aggregate scoring event and return its validated request ID."""
    errors = Counter(row.error_code for row in result.rows if row.error_code)
    return log_score_summary(
        model_version=result.summary.model_version,
        total_rows=result.summary.total_rows,
        succeeded_rows=result.summary.succeeded_rows,
        failed_rows=result.summary.failed_rows,
        elapsed_ms=elapsed_ms,
        error_code_counts=dict(errors),
        request_id=request_id,
    )
