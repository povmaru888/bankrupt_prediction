"""In-memory, request-session state for the four Streamlit pages."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

SCORE_RESPONSE_KEY = "score_response"
UPLOAD_BYTES_KEY = "uploaded_csv_bytes"
UPLOAD_FILENAME_KEY = "uploaded_filename"
SESSION_KEYS = (SCORE_RESPONSE_KEY, UPLOAD_BYTES_KEY, UPLOAD_FILENAME_KEY)


def save_score_session(
    state: MutableMapping[str, Any], payload: dict[str, Any], csv_bytes: bytes, filename: str
) -> None:
    state[SCORE_RESPONSE_KEY] = payload
    state[UPLOAD_BYTES_KEY] = bytes(csv_bytes)
    state[UPLOAD_FILENAME_KEY] = filename


def get_score_session(state: MutableMapping[str, Any]) -> tuple[dict[str, Any], bytes, str] | None:
    if not all(key in state for key in SESSION_KEYS):
        return None
    payload = state[SCORE_RESPONSE_KEY]
    data = state[UPLOAD_BYTES_KEY]
    filename = state[UPLOAD_FILENAME_KEY]
    if (
        not isinstance(payload, dict)
        or not isinstance(data, bytes)
        or not isinstance(filename, str)
    ):
        return None
    return payload, data, filename


def clear_score_session(state: MutableMapping[str, Any]) -> None:
    for key in SESSION_KEYS:
        state.pop(key, None)
