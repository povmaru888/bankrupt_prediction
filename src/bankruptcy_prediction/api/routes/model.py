"""Read-only active model summary endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankruptcy_prediction.api.routes.health import resolve_active_model
from bankruptcy_prediction.model_registry.registry import ActiveModelUnavailable

router = APIRouter(prefix="/api/v1")


@router.get("/model")
async def model_info(request: Request) -> dict[str, object]:
    try:
        model = resolve_active_model(request)
    except ActiveModelUnavailable:
        raise
    return model.summary.to_dict()
