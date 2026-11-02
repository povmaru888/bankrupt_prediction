"""Human-invoked model activation and rollback commands."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from bankruptcy_prediction.model_package.verifier import VerifiedPackage, verify_package

VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RegistryCommandError(ValueError):
    """Raised when a version cannot safely become the active model."""


@dataclass(frozen=True)
class ActivationResult:
    action: Literal["activate", "rollback"]
    version: str
    previous_version: str | None
    active_path: Path
    updated_at: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "action": self.action,
            "version": self.version,
            "previous_version": self.previous_version,
            "active_path": str(self.active_path),
            "updated_at": self.updated_at,
        }


def _read_active(active_path: Path) -> str | None:
    if active_path.is_symlink():
        raise RegistryCommandError(
            "active model pointer is a symbolic link; refusing to overwrite it"
        )
    if not active_path.exists():
        return None
    try:
        pointer = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryCommandError(
            "active model pointer is invalid; refusing to overwrite it"
        ) from exc
    if not isinstance(pointer, dict) or not isinstance(pointer.get("version"), str):
        raise RegistryCommandError("active model pointer is invalid; refusing to overwrite it")
    return pointer["version"]


def _resolve_package(models_root: Path, version: str) -> VerifiedPackage:
    if (
        not isinstance(version, str)
        or not VERSION_PATTERN.fullmatch(version)
        or version in {".", ".."}
    ):
        raise RegistryCommandError("model version identifier is invalid")
    root = models_root.resolve()
    package_path = root / version
    try:
        package_path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise RegistryCommandError(
            "model version is unavailable or outside the models directory"
        ) from exc
    try:
        return verify_package(package_path)
    except ValueError as exc:
        raise RegistryCommandError("model version failed package verification") from exc


def _set_active(
    action: Literal["activate", "rollback"], version: str, models_root: str | Path
) -> ActivationResult:
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    # Verify every artifact before writing the pointer; verification never unpickles the model.
    package = _resolve_package(root, version)
    active_path = root / "active.json"
    previous_version = _read_active(active_path)
    updated_at = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": 1,
        "version": package.version,
        "previous_version": previous_version,
        "action": action,
        "updated_at": updated_at,
    }
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".active-", suffix=".tmp", dir=root)
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, active_path)
    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, OSError):
            raise RegistryCommandError(
                "active model pointer update failed; previous value preserved"
            ) from exc
        raise
    return ActivationResult(
        action=action,
        version=package.version,
        previous_version=previous_version,
        active_path=active_path,
        updated_at=updated_at,
    )


def activate_model(version: str, models_root: str | Path) -> ActivationResult:
    """Validate and set an existing candidate as active without changing its package."""
    return _set_active("activate", version, models_root)


def rollback_model(version: str, models_root: str | Path) -> ActivationResult:
    """Validate and switch to an explicitly requested existing model version."""
    return _set_active("rollback", version, models_root)
