"""Small typed HTTP client for the localhost FastAPI service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


class ApiClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


@dataclass(frozen=True)
class ApiClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        configured = os.environ.get("BANKRUPTCY_API_BASE_URL")
        if configured and self.base_url == "http://127.0.0.1:8000":
            object.__setattr__(self, "base_url", configured)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self.base_url.rstrip('/')}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ApiClientError("無法連線至本機 API，請確認 FastAPI 已啟動。") from exc
        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            message = (
                (payload or {}).get("message", "API 請求失敗。")
                if isinstance(payload, dict)
                else "API 請求失敗。"
            )
            raise ApiClientError(message, status_code=response.status_code, payload=payload)
        return response

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def model(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/model").json()

    def score(
        self,
        csv_bytes: bytes,
        *,
        filename: str = "input.csv",
        threshold: float | None = None,
        response_format: str = "json",
    ) -> Any:
        params: dict[str, Any] = {"format": response_format}
        if threshold is not None:
            params["threshold"] = threshold
        response = self._request(
            "POST",
            "/api/v1/score",
            params=params,
            files={"file": (filename, csv_bytes, "text/csv")},
        )
        return response.json() if response_format == "json" else response.content
