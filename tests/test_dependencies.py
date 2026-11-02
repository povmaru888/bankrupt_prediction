"""Verify that the runtime and development requirements stay pinned."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            pins.update(_pins(path.parent / line[3:].strip()))
            continue
        name, separator, version = line.partition("==")
        assert separator, f"Dependency is not pinned with ==: {line}"
        pins[re.sub(r"[-_.]+", "-", name.lower())] = version
    return pins


def test_runtime_and_development_dependencies_are_pinned_and_installed() -> None:
    runtime = _pins(ROOT / "requirements.txt")
    development = _pins(ROOT / "requirements-dev.txt")

    assert {
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "shap",
        "joblib",
        "matplotlib",
        "fastapi",
        "pydantic",
        "uvicorn",
        "python-multipart",
        "streamlit",
        "pyyaml",
    } <= runtime.keys()
    assert {"pytest", "pytest-cov", "ruff", "httpx"} <= development.keys()
    assert runtime.items() <= development.items()

    for name, expected_version in development.items():
        assert importlib.metadata.version(name) == expected_version
