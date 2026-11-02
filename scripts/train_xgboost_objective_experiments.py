"""Train rank and Pseudo-Huber XGBoost experiments without touching final test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from bankruptcy_prediction.data.feature_schema import build_feature_schema
from bankruptcy_prediction.data.training_loader import load_training_data
from bankruptcy_prediction.features.imputation import build_imputer
from bankruptcy_prediction.features.selector import FeatureSelector
from bankruptcy_prediction.model_package.verifier import verify_package
from bankruptcy_prediction.training.objective_experiments import ObjectiveExperimentModel
from bankruptcy_prediction.training.split_manager import create_split

SEED = 42
ITERATIONS = 30
FOLDS = 5


def pipeline(kind: str) -> Pipeline:
    return Pipeline(
        [
            ("selector", FeatureSelector()),
            ("imputer", build_imputer()),
            ("classifier", ObjectiveExperimentModel(objective_kind=kind)),
        ]
    )


def search_space(kind: str) -> dict[str, Any]:
    space: dict[str, Any] = {
        "classifier__max_depth": randint(3, 10),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__n_estimators": randint(100, 1001),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 11),
        "classifier__reg_alpha": loguniform(1e-8, 10.0),
        "classifier__reg_lambda": loguniform(1e-3, 100.0),
    }
    if kind == "rank_pairwise":
        space["classifier__lambdarank_num_pair_per_sample"] = randint(4, 33)
    else:
        space["classifier__huber_slope"] = loguniform(0.1, 10.0)
    return space


def top_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    count = int(np.ceil(len(y) * 0.10))
    order = np.argsort(-scores, kind="stable")[:count]
    positives = int(y.sum())
    hits = int(y[order].sum())
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "recall_at_top_10_percent": hits / positives,
        "precision_at_top_10_percent": hits / count,
        "ndcg_at_top_10_percent": float(ndcg_score(y[None, :], scores[None, :], k=count)),
        "top_10_percent_row_count": count,
        "top_10_percent_positive_count": hits,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ExperimentResult:
    kind: str
    estimator: Any
    search_cv_ap: float
    best_parameters: dict[str, Any]
    validation_metrics: dict[str, float | int]
    elapsed_seconds: float


def train(
    kind: str, X: Any, y: Any, X_validation: Any, y_validation: np.ndarray
) -> ExperimentResult:
    print(f"START {kind}: {ITERATIONS} candidates x {FOLDS} folds", flush=True)
    start = time.monotonic()
    search = RandomizedSearchCV(
        pipeline(kind),
        search_space(kind),
        n_iter=ITERATIONS,
        scoring="average_precision",
        n_jobs=1,
        cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED),
        random_state=SEED,
        refit=True,
        error_score="raise",
        verbose=2,
    )
    search.fit(X, y)
    scores = search.best_estimator_.predict_proba(X_validation)[:, 1]
    result = ExperimentResult(
        kind=kind,
        estimator=search.best_estimator_,
        search_cv_ap=float(search.best_score_),
        best_parameters=dict(search.best_params_),
        validation_metrics=top_metrics(y_validation, scores),
        elapsed_seconds=time.monotonic() - start,
    )
    print(
        json.dumps(
            {
                "kind": kind,
                "search_cv_ap": result.search_cv_ap,
                "validation_metrics": result.validation_metrics,
                "elapsed_seconds": result.elapsed_seconds,
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--binary-package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_path = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    dataset = load_training_data(data_path)
    dataset.features.columns = [field.name for field in build_feature_schema(dataset).features]
    split = create_split(dataset.labels.to_numpy(), seed=SEED)
    X_train = dataset.features.iloc[list(split.train_indices)]
    y_train = dataset.labels.iloc[list(split.train_indices)]
    X_validation = dataset.features.iloc[list(split.validation_indices)]
    y_validation = dataset.labels.iloc[list(split.validation_indices)].to_numpy(dtype=int)

    verified = verify_package(args.binary_package)
    binary_model = joblib.load(verified.model_path)
    binary_scores = binary_model.predict_proba(X_validation)[:, 1]
    comparison: dict[str, Any] = {
        "protocol": {
            "seed": SEED,
            "search_iterations": ITERATIONS,
            "cv_folds": FOLDS,
            "selection_partition": "validation",
            "final_test_accessed": False,
            "data_sha256": sha256(data_path),
            "rank_grouping": "one query per fit because the source has no query ID",
            "huber_weighting": "fold-local negative/positive balanced sample weights",
        },
        "binary_logistic": {
            "source_package": verified.version,
            "validation_metrics": top_metrics(y_validation, binary_scores),
        },
    }
    for kind in ("rank_pairwise", "pseudo_huber"):
        result = train(kind, X_train, y_train, X_validation, y_validation)
        model_path = output / f"xgboost_{kind}.joblib"
        joblib.dump(result.estimator, model_path)
        comparison[kind] = {
            "search_cv_average_precision": result.search_cv_ap,
            "best_parameters": result.best_parameters,
            "validation_metrics": result.validation_metrics,
            "elapsed_seconds": result.elapsed_seconds,
            "model_file": model_path.name,
            "model_sha256": sha256(model_path),
        }
    comparison["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "COMPLETED").write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
