"""FastAPI app factory with localhost-safe lifecycle and error handling."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

from bankruptcy_prediction.api.error_handlers import (
    handle_boundary_error,
    handle_file_schema_error,
    handle_model_unavailable,
    handle_record_id_error,
    handle_request_validation,
    handle_response_format_error,
    handle_unexpected_error,
)
from bankruptcy_prediction.api.errors import ApiBoundaryError
from bankruptcy_prediction.api.routes import health, model, score
from bankruptcy_prediction.data.record_ids import RecordIdError
from bankruptcy_prediction.data.schema_validator import FileSchemaError
from bankruptcy_prediction.model_registry.registry import ActiveModelUnavailable
from bankruptcy_prediction.scoring.response_formatter import ResponseFormatError
from bankruptcy_prediction.settings import AppSettings, load_settings


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_runtime_settings(project_root: str | Path | None = None) -> AppSettings:
    """Load local settings and anchor every service directory to the project root."""
    root = Path(project_root).resolve() if project_root is not None else _project_root()
    settings = load_settings(root / "config" / "training.yaml", root / "config" / "service.yaml")
    service = settings.service
    anchored_service = replace(
        service,
        models_dir=str((root / service.models_dir).resolve()),
        logs_dir=str((root / service.logs_dir).resolve()),
        temp_dir=str((root / service.temp_dir).resolve()),
    )
    return replace(settings, service=anchored_service)


def create_app(settings: Any, *, model_loader: Callable[[str], Any] | None = None) -> FastAPI:
    """Create an API instance without loading or mutating a model at import time."""
    app = FastAPI(title="企業破產預測系統 API", version="0.1.0")
    app.state.settings = settings
    app.state.model_loader = model_loader
    app.state.active_model = None

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "").strip()
        request_id = (
            supplied[:128]
            if supplied and all(ord(char) >= 32 for char in supplied)
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.add_exception_handler(ApiBoundaryError, handle_boundary_error)
    app.add_exception_handler(FileSchemaError, handle_file_schema_error)
    app.add_exception_handler(RecordIdError, handle_record_id_error)
    app.add_exception_handler(ResponseFormatError, handle_response_format_error)
    app.add_exception_handler(ActiveModelUnavailable, handle_model_unavailable)
    app.add_exception_handler(Exception, handle_unexpected_error)
    from fastapi.exceptions import RequestValidationError

    app.add_exception_handler(RequestValidationError, handle_request_validation)
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(score.router)

    @app.get("/__app_ready")
    async def app_ready() -> dict[str, str]:
        return {"status": "ok"}

    return app


def create_runtime_app() -> FastAPI:
    """Zero-argument Uvicorn factory for the local production process."""
    return create_app(load_runtime_settings())
