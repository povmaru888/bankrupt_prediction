"""Explicit regression tests that selection and evaluation never fit on test data."""

import numpy as np
import pandas as pd

from bankruptcy_prediction.evaluation.metrics import evaluate_frozen_model
from bankruptcy_prediction.features.imputation import build_imputer
from bankruptcy_prediction.features.range_monitor import fit_ranges
from bankruptcy_prediction.training.threshold_selector import select_f2_threshold


def test_imputer_and_range_use_train_statistics_only():
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [10.0, 30.0]})
    test = pd.DataFrame({"a": [1000.0], "b": [-1000.0]})
    imputer = build_imputer().fit(train)
    ranges = fit_ranges(train)
    assert imputer.statistics_.tolist() == [2.0, 20.0]
    transformed = imputer.transform(test)
    assert transformed.tolist() == [[1000.0, -1000.0]]
    warnings = ranges.flag_out_of_range(test)
    assert warnings.out_of_range_count.tolist() == [2]


def test_threshold_selection_is_validation_only_and_final_evaluation_is_frozen():
    validation = select_f2_threshold([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2])
    assert 0 <= validation.default_threshold <= 1

    class FrozenModel:
        def __init__(self):
            self.fit_called = False

        def predict_proba(self, X):
            assert len(X) == 4
            return np.column_stack([1 - np.asarray(X), np.asarray(X)])

        def fit(self, *args, **kwargs):
            self.fit_called = True
            raise AssertionError("final-test evaluation must not fit")

    model = FrozenModel()
    report = evaluate_frozen_model(
        model, validation.default_threshold, [0.1, 0.8, 0.7, 0.2], [0, 1, 1, 0]
    )
    assert report.sample_count == 4
    assert model.fit_called is False
