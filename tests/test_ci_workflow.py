"""Static checks for the CI quality gates and ignored local artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_runs_install_check_format_lint_and_tests_in_order() -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    triggers = workflow["on"]
    steps = workflow["jobs"]["quality"]["steps"]
    names = [step.get("name", "") for step in steps]

    assert triggers == ["push", "pull_request"]
    assert "Install dependencies" in names[2]
    assert "Check dependency consistency" in names[3]
    assert "Check formatting" in names[4]
    assert "Lint" in names[5]
    assert "Run tests" in names[6]
    assert steps[2]["run"].find("requirements-dev.txt") < steps[2]["run"].find(
        "pip install --no-deps -e ."
    )
    assert steps[3]["run"] == "python -m pip check"
    assert steps[4]["run"] == "ruff format --check src tests"
    assert steps[5]["run"] == "ruff check src tests"
    assert steps[6]["run"] == "pytest -q"


def test_gitignore_excludes_local_and_binary_artifact_paths() -> None:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for required_rule in (
        ".venv/",
        "tmp/uploads/",
        "models/**/*.joblib",
        "models/active.json",
        "logs/*",
        "!logs/.gitkeep",
    ):
        assert required_rule in rules


def test_ruff_config_targets_supported_python_version() -> None:
    config = (ROOT / "ruff.toml").read_text(encoding="utf-8")

    assert 'target-version = "py311"' in config
    assert "line-length = 100" in config
