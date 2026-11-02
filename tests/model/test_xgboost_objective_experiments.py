import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_xgboost_objective_experiments import (  # noqa: E402
    ObjectiveExperimentModel,
    top_metrics,
)


def test_rank_and_huber_adapters_fit_and_emit_finite_monotonic_scores():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 4))
    y = np.array([0] * 30 + [1] * 10)
    for kind in ("rank_pairwise", "pseudo_huber"):
        model = ObjectiveExperimentModel(
            objective_kind=kind,
            n_estimators=3,
            max_depth=2,
            lambdarank_num_pair_per_sample=4,
        ).fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (40, 2)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_top_metrics_use_stable_top_ten_percent():
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    metrics = top_metrics(y, scores)
    assert metrics["top_10_percent_row_count"] == 1
    assert metrics["recall_at_top_10_percent"] == 0.5
    assert metrics["precision_at_top_10_percent"] == 1.0
