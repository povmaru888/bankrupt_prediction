"""One-time migration from __main__ pickle references to the stable model module."""

import hashlib
import json
from pathlib import Path

import joblib

import __main__
from bankruptcy_prediction.training.objective_experiments import ObjectiveExperimentModel

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/xgboost-objectives-20261022"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__main__.ObjectiveExperimentModel = ObjectiveExperimentModel
comparison_path = OUTPUT / "comparison.json"
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
for key in ("rank_pairwise", "pseudo_huber"):
    path = OUTPUT / comparison[key]["model_file"]
    model = joblib.load(path)
    joblib.dump(model, path)
    comparison[key]["model_sha256"] = sha256(path)
comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print("Repackaged both experiment models with stable module references.")
