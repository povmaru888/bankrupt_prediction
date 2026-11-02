"""Model reproducibility and MVP quality-gate acceptance checks."""

from pathlib import Path

import numpy as np

from bankruptcy_prediction.evaluation.metrics import evaluate_frozen_scores
from bankruptcy_prediction.settings import load_settings
from bankruptcy_prediction.training.model_search import EXPECTED_FOLDS, EXPECTED_ITERATIONS

ROOT = Path(__file__).resolve().parents[2]


def test_official_settings_encode_reproducible_search_baseline():
    settings = load_settings(ROOT / "config/training.yaml", ROOT / "config/service.yaml").training
    assert settings.random_seed == 42
    assert settings.split.train == 0.64
    assert settings.split.validation == 0.16
    assert settings.split.test == 0.20
    assert settings.search.n_iter == EXPECTED_ITERATIONS == 30
    assert settings.search.cv_folds == EXPECTED_FOLDS == 5
    assert settings.search.scoring == "average_precision"


def test_quality_gate_reports_both_mvp_requirements_without_tuning_test():
    labels = np.array([1, 1] + [0] * 18)
    scores = np.array([0.99, 0.9] + [0.1] * 18)
    report = evaluate_frozen_scores(0.5, labels, scores)
    assert report.sample_count == len(labels)
    assert report.threshold == 0.5
    assert report.ap_to_prevalence_ratio is not None
    assert report.recall_at_top_10_percent >= 0.5
