"""Project metadata and package layout checks."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_and_package_root() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "bankruptcy-prediction"
    assert metadata["project"]["requires-python"] == ">=3.11"
    assert metadata["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert metadata["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert (ROOT / "src" / "bankruptcy_prediction" / "__init__.py").is_file()
