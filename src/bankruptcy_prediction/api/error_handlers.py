"""Consistent, non-sensitive FastAPI exception responses."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from bankruptcy_prediction.api.errors import ApiBoundaryError
from bankruptcy_prediction.data.record_ids import RecordIdError
from bankruptcy_prediction.data.schema_validator import FileSchemaError
from bankruptcy_prediction.model_registry.registry import ActiveModelUnavailable
from bankruptcy_prediction.scoring.response_formatter import ResponseFormatError

LOGGER = logging.getLogger(__name__)


def error_payload(
    error_code: str,
    message: str,
    details: dict[str, Any] | None,
    remediation: str,
    request: Request,
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "remediation": remediation,
        "request_id": getattr(request.state, "request_id", ""),
    }


def _response(request: Request, status_code: int, **kwargs: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(request=request, **kwargs))


async def handle_boundary_error(request: Request, exc: ApiBoundaryError) -> JSONResponse:
    return _response(
        request,
        exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        remediation=exc.remediation,
    )


async def handle_file_schema_error(request: Request, exc: FileSchemaError) -> JSONResponse:
    return _response(
        request,
        422,
        error_code=exc.api_error.error_code,
        message=exc.api_error.message,
        details=exc.api_error.details,
        remediation=exc.api_error.remediation,
    )


async def handle_record_id_error(request: Request, exc: RecordIdError) -> JSONResponse:
    return _response(
        request,
        422,
        error_code="RECORD_ID_INVALID",
        message="record_id 必須非空且在批次內唯一。",
        details={},
        remediation="請修正 record_id 後重試。",
    )


async def handle_response_format_error(request: Request, exc: ResponseFormatError) -> JSONResponse:
    return _response(
        request,
        422,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        remediation=exc.remediation,
    )


async def handle_model_unavailable(request: Request, exc: ActiveModelUnavailable) -> JSONResponse:
    return _response(
        request,
        503,
        error_code=exc.error_code,
        message="active model 目前不可用。",
        details={},
        remediation="請檢查模型包並由 CLI 啟用模型。",
    )


async def handle_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _response(
        request,
        422,
        error_code="REQUEST_VALIDATION_FAILED",
        message="請求格式驗證失敗。",
        details={"fields": [str(item.get("loc", ())) for item in exc.errors()]},
        remediation="請依 API 欄位格式修正請求。",
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("unhandled API exception", exc_info=exc)
    return _response(
        request,
        500,
        error_code="INTERNAL_ERROR",
        message="服務處理請求時發生未預期錯誤。",
        details={},
        remediation="請稍後重試，並提供 request_id 給維運人員。",
    )
