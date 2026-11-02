"""Service health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankruptcy_prediction.model_registry.registry import ActiveModelUnavailable, load_active_model

router = APIRouter()


def resolve_active_model(request: Request):
    loader = getattr(request.app.state, "model_loader", None)
    if loader is not None:
        return loader(request.app.state.settings.service.models_dir)
    return load_active_model(request.app.state.settings.service.models_dir)


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    try:
        model = resolve_active_model(request)
    except ActiveModelUnavailable as exc:
        return {
            "status": "degraded",
            "active_model_loaded": False,
            "reason_code": exc.error_code,
        }
    return {
        "status": "ok",
        "active_model_loaded": True,
        "model_version": model.version,
    }
