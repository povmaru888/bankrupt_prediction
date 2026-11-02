"""Tests for unique request workspaces and deterministic temporary-file cleanup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bankruptcy_prediction.service.temp_files import (
    WORKSPACE_PREFIX,
    cleanup_stale_upload_workspaces,
    create_upload_file_path,
    temporary_upload_workspace,
)


def test_workspace_cleans_after_success_validation_error_and_unexpected_error(
    tmp_path: Path,
) -> None:
    for exception in (None, ValueError("invalid CSV"), RuntimeError("unexpected")):
        workspace_path: Path | None = None
        if exception is None:
            with temporary_upload_workspace(tmp_path) as workspace:
                workspace_path = workspace
                create_upload_file_path(workspace, "normal.csv").write_text("a,b\n1,2\n")
        else:
            with pytest.raises(type(exception)):
                with temporary_upload_workspace(tmp_path) as workspace:
                    workspace_path = workspace
                    create_upload_file_path(workspace, "../../outside.csv").write_text("bad")
                    raise exception
        assert workspace_path is not None
        assert not workspace_path.exists()


def test_untrusted_or_duplicate_filenames_never_control_paths(tmp_path: Path) -> None:
    with temporary_upload_workspace(tmp_path) as workspace:
        first = create_upload_file_path(workspace, "../../same.csv")
        second = create_upload_file_path(workspace, "../../same.csv")

        assert first != second
        assert first.parent == workspace
        assert second.parent == workspace
        assert first.name.startswith("upload-") and first.suffix == ".csv"
        assert first.is_file() and second.is_file()


def test_cleanup_removes_only_old_managed_workspaces(tmp_path: Path) -> None:
    old = tmp_path / f"{WORKSPACE_PREFIX}old"
    fresh = tmp_path / f"{WORKSPACE_PREFIX}fresh"
    unrelated = tmp_path / "keep-me"
    old.mkdir()
    fresh.mkdir()
    unrelated.mkdir()
    os.utime(old, (10, 10))
    os.utime(fresh, (95, 95))

    removed = cleanup_stale_upload_workspaces(tmp_path, max_age_seconds=50, now=100)

    assert removed == 1
    assert not old.exists()
    assert fresh.is_dir() and unrelated.is_dir()


def test_rejects_unmanaged_workspace_and_invalid_sweep_age(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="managed upload directory"):
        create_upload_file_path(tmp_path, "../../escape.csv")
    with pytest.raises(ValueError, match="positive integer"):
        cleanup_stale_upload_workspaces(tmp_path, max_age_seconds=0)
