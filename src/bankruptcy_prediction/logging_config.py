"""Privacy-preserving structured scoring audit logs."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from bankruptcy_prediction.settings import AppSettings

_LOGGER_NAME = "bankruptcy_prediction.audit"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "audit_payload", None)
        if not isinstance(payload, dict):
            return "{}"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _safe_token(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(f"{field} must contain only letters, digits, dot, underscore, or hyphen")
    return value


def configure_logging(settings: AppSettings) -> None:
    """Configure an append-only JSONL audit file under the configured logs directory."""
    log_dir = Path(settings.service.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / "scoring_audit.jsonl").resolve()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(handler)


def log_score_summary(
    *,
    model_version: str,
    total_rows: int,
    succeeded_rows: int,
    failed_rows: int,
    elapsed_ms: int,
    error_code_counts: Mapping[str, int],
    request_id: str | None = None,
) -> str:
    """Write only aggregate scoring metadata and return its request correlation ID."""
    model_version = _safe_token(model_version, "model_version")
    request_id = _safe_token(request_id or uuid.uuid4().hex, "request_id")
    for name, value in (
        ("total_rows", total_rows),
        ("succeeded_rows", succeeded_rows),
        ("failed_rows", failed_rows),
        ("elapsed_ms", elapsed_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if succeeded_rows + failed_rows != total_rows:
        raise ValueError("succeeded_rows + failed_rows must equal total_rows")

    safe_errors: dict[str, int] = {}
    for code, count in error_code_counts.items():
        safe_code = _safe_token(code, "error_code")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("error code counts must be non-negative integers")
        safe_errors[safe_code] = count

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "total_rows": total_rows,
        "succeeded_rows": succeeded_rows,
        "failed_rows": failed_rows,
        "elapsed_ms": elapsed_ms,
        "error_code_counts": safe_errors,
    }
    logging.getLogger(_LOGGER_NAME).info("scoring_summary", extra={"audit_payload": payload})
    return request_id
