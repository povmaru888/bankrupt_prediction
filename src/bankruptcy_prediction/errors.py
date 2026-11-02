"""Shared machine-readable error contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiError:
    """Stable API error envelope shared by CLI, API, and UI layers."""

    error_code: str
    message: str
    details: dict[str, Any]
    remediation: str

    def __post_init__(self) -> None:
        for name in ("error_code", "message", "remediation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.details, dict):
            raise TypeError("details must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ApiError:
        return cls(
            error_code=payload["error_code"],
            message=payload["message"],
            details=payload["details"],
            remediation=payload["remediation"],
        )
