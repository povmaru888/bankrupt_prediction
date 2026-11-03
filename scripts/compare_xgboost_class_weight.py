"""Run the explicitly authorized XGBoost class-weight ablation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from bankruptcy_prediction.training.class_weight_comparison import (
    evaluate_class_weight_comparison,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root / "Database" / "bankrupt_data.csv"))
    parser.add_argument(
        "--candidate",
        default=str(root / "models" / "candidate-20261022T105726Z-155ceb0b"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "experiments" / "xgboost-class-weight-20261103"),
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_class_weight_comparison(
        args.data,
        args.candidate,
        output / "xgboost_unweighted.joblib",
    )
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "COMPLETED").write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    print(comparison_path)


if __name__ == "__main__":
    main()
