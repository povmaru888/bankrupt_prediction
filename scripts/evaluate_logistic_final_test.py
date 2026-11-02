"""Generate the explicitly authorized post-hoc LR vs XGBoost comparison artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bankruptcy_prediction.training.posthoc_comparison import evaluate_posthoc_logistic


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(root / "Database" / "bankrupt_data.csv"))
    parser.add_argument(
        "--candidate",
        default=str(root / "models" / "candidate-20261022T105726Z-155ceb0b"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "experiments" / "posthoc-final-test-comparison.json"),
    )
    parser.add_argument(
        "--chart-output",
        default=str(
            root
            / "experiments"
            / "posthoc-final-test-comparison"
            / "precision_recall_comparison.png"
        ),
    )
    parser.add_argument(
        "--confusion-chart-output",
        default=str(
            root / "experiments" / "posthoc-final-test-comparison" / "confusion_matrix_xgboost.png"
        ),
    )
    args = parser.parse_args()
    result = evaluate_posthoc_logistic(
        args.data,
        args.candidate,
        chart_output=args.chart_output,
        confusion_chart_output=args.confusion_chart_output,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
