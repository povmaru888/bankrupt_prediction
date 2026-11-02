"""Multipart CSV scoring endpoint."""

from __future__ import annotations

from io import BytesIO
from time import perf_counter

import pandas as pd
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from bankruptcy_prediction.api.errors import ApiBoundaryError, PayloadTooLargeError
from bankruptcy_prediction.api.routes.health import resolve_active_model
from bankruptcy_prediction.scoring.response_formatter import (
    CsvResponse,
    JsonResponse,
    format_score_response,
)
from bankruptcy_prediction.scoring.service import score_batch
from bankruptcy_prediction.service.audit_logger import record_score_request
from bankruptcy_prediction.service.temp_files import (
    create_upload_file_path,
    temporary_upload_workspace,
)

router = APIRouter(prefix="/api/v1")


async def _read_bounded(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError("上傳檔案超過大小限制。", details={"max_bytes": limit})
        chunks.append(chunk)
    return b"".join(chunks)


def _response(formatted: JsonResponse | CsvResponse) -> Response:
    if isinstance(formatted, JsonResponse):
        return JSONResponse(content=formatted.body, media_type=formatted.media_type)
    return Response(
        content=formatted.body,
        media_type=formatted.media_type,
        headers={"Content-Disposition": f'attachment; filename="{formatted.filename}"'},
    )


@router.post("/score")
async def score(
    request: Request,
    file: UploadFile = File(...),
    format: str = Query("json", pattern="^(json|csv)$"),
    threshold: float | None = Query(None),
) -> Response:
    settings = request.app.state.settings.service
    limit = int(settings.max_upload_size_mb) * 1024 * 1024
    raw = await _read_bounded(file, limit)
    if not raw:
        raise ApiBoundaryError("上傳檔案不可為空。", details={"field": "file"})
    try:
        with temporary_upload_workspace(settings.temp_dir) as workspace:
            upload_path = create_upload_file_path(workspace, file.filename)
            upload_path.write_bytes(raw)
            frame = pd.read_csv(BytesIO(raw))
    except PayloadTooLargeError:
        raise
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ApiBoundaryError("CSV 檔案無法解析。", details={"reason": "invalid_csv"}) from exc
    if len(frame) > int(settings.max_rows):
        raise PayloadTooLargeError(
            "CSV 列數超過限制。", details={"max_rows": int(settings.max_rows)}
        )
    active_model = resolve_active_model(request)
    started = perf_counter()
    result = score_batch(active_model, frame, threshold=threshold)
    record_score_request(
        result,
        elapsed_ms=max(0, int((perf_counter() - started) * 1000)),
        request_id=request.state.request_id,
    )
    return _response(format_score_response(result, format, threshold=threshold))
