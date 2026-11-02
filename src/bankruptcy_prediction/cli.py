"""Command-line entry point for offline candidate model training."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bankruptcy_prediction.model_registry.commands import activate_model, rollback_model
from bankruptcy_prediction.settings import load_settings
from bankruptcy_prediction.training.orchestrator import run_training


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bankruptcy-prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and package a model candidate")
    train.add_argument("--data", required=True, help="path to the audited training CSV")
    train.add_argument("--config", required=True, help="path to the training YAML config")
    train.add_argument(
        "--service-config",
        default=str(_project_root() / "config" / "service.yaml"),
        help="path to service YAML (used for shared settings validation)",
    )
    train.add_argument(
        "--models-root",
        default=str(_project_root() / "models"),
        help="directory where immutable candidate packages are stored",
    )
    for action in ("activate", "rollback"):
        command = commands.add_parser(action, help=f"{action} a verified model version")
        command.add_argument("--version", required=True, help="existing model version ID")
        command.add_argument(
            "--models-root",
            default=str(_project_root() / "models"),
            help="directory containing versioned model packages",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            settings = load_settings(args.config, args.service_config)
            outcome = run_training(args.data, settings.training, args.models_root)
            result = outcome.to_dict()
        elif args.command == "activate":
            result = activate_model(args.version, args.models_root).to_dict()
        else:
            result = rollback_model(args.version, args.models_root).to_dict()
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
