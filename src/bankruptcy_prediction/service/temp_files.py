"""Secure request-scoped upload workspaces with bounded startup cleanup."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

WORKSPACE_PREFIX = "bankruptcy-upload-"
UPLOAD_PREFIX = "upload-"
WORKSPACE_MAX_AGE_SECONDS = 24 * 60 * 60
logger = logging.getLogger(__name__)


@contextmanager
def temporary_upload_workspace(temp_root: str | Path | None = None) -> Iterator[Path]:
    """Yield a cryptographically random workspace and attempt cleanup on every exit."""
    root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve(strict=True)
    workspace = Path(tempfile.mkdtemp(prefix=WORKSPACE_PREFIX, dir=resolved_root))
    try:
        yield workspace
    finally:
        try:
            shutil.rmtree(workspace)
        except FileNotFoundError:
            pass
        except OSError:
            # Do not log local paths or uploaded filenames; startup cleanup handles stale dirs.
            logger.error("temporary upload workspace cleanup failed")


def create_upload_file_path(workspace: str | Path, original_filename: str | None = None) -> Path:
    """Reserve a unique server-named CSV file; the user filename is never used as a path."""
    del original_filename
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or not root.name.startswith(WORKSPACE_PREFIX):
        raise ValueError("workspace must be a managed upload directory")
    descriptor, filename = tempfile.mkstemp(prefix=UPLOAD_PREFIX, suffix=".csv", dir=root)
    os.close(descriptor)
    result = Path(filename).resolve(strict=True)
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError("temporary upload path escaped its workspace") from exc
    return result


def cleanup_stale_upload_workspaces(
    temp_root: str | Path,
    *,
    max_age_seconds: int = WORKSPACE_MAX_AGE_SECONDS,
    now: float | None = None,
) -> int:
    """Remove only expired managed upload directories at startup; never scan recursively."""
    if (
        isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, int)
        or max_age_seconds < 1
    ):
        raise ValueError("max_age_seconds must be a positive integer")
    root = Path(temp_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("temp_root must be a directory")
    cutoff = (time.time() if now is None else now) - max_age_seconds
    cleaned = 0
    for entry in root.iterdir():
        if not entry.name.startswith(WORKSPACE_PREFIX) or entry.is_symlink() or not entry.is_dir():
            continue
        try:
            if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                shutil.rmtree(entry)
                cleaned += 1
        except OSError:
            logger.error("stale temporary upload workspace cleanup failed")
    return cleaned
