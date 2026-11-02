"""HTTP-facing exceptions used by the local API boundary."""

from __future__ import annotations

from typing import Any


class ApiBoundaryError(Exception):
    status_code = 422
    error_code = "REQUEST_INVALID"
    remediation = "請修正請求後重試。"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class PayloadTooLargeError(ApiBoundaryError):
    status_code = 413
    error_code = "PAYLOAD_TOO_LARGE"
    remediation = "請縮小檔案或列數後重試。"


class NoActiveModelError(ApiBoundaryError):
    status_code = 503
    error_code = "NO_ACTIVE_MODEL"
    remediation = "請先由離線 CLI 啟用可驗證的模型。"
