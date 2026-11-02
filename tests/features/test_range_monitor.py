"""Tests for training-only range baselines and out-of-range warnings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from bankruptcy_prediction.contracts import WarningCode
from bankruptcy_prediction.features.range_monitor import (
    MAX_REPORTED_COLUMNS_PER_ROW,
    RangeMonitor,
    fit_ranges,
)


def test_training_bounds_allow_exact_edges_without_warning() -> None:
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [-2.0, 2.0]})
    monitor = fit_ranges(train)
    edges = pd.DataFrame({"a": [1.0, 3.0], "b": [-2.0, 2.0]})

    warnings = monitor.flag_out_of_range(edges)

    assert warnings.warning_codes == ((), ())
    assert warnings.out_of_range_count.tolist() == [0, 0]
    assert warnings.out_of_range_columns == ((), ())


def test_finite_out_of_range_values_are_flagged_but_not_clipped() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0], "b": [10.0, 20.0]})
    inference = pd.DataFrame({"a": [-1000.0], "b": [2000.0]})
    monitor = fit_ranges(train)

    warnings = monitor.flag_out_of_range(inference)

    assert warnings.warning_codes == ((WarningCode.OUT_OF_RANGE.value,),)
    assert warnings.out_of_range_columns == (("a", "b"),)
    assert warnings.out_of_range_count.tolist() == [2]
    assert inference.iloc[0].tolist() == [-1000.0, 2000.0]
    assert monitor.minimum_.tolist() == [0.0, 10.0]
    assert monitor.maximum_.tolist() == [1.0, 20.0]


def test_multi_column_summary_is_bounded_but_count_is_complete() -> None:
    columns = [f"feature_{index}" for index in range(15)]
    train = pd.DataFrame([[0.0] * 15, [1.0] * 15], columns=columns)
    inference = pd.DataFrame([[2.0] * 15], columns=columns)

    warnings = fit_ranges(train).flag_out_of_range(inference)

    assert len(warnings.out_of_range_columns[0]) == MAX_REPORTED_COLUMNS_PER_ROW
    assert warnings.out_of_range_count.iloc[0] == 15
    assert warnings.out_of_range_columns[0] == tuple(columns[:MAX_REPORTED_COLUMNS_PER_ROW])


def test_rejects_infinite_values_and_transform_before_fit() -> None:
    with pytest.raises(NotFittedError):
        RangeMonitor().flag_out_of_range([[1.0]])

    monitor = fit_ranges(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="cannot contain infinite"):
        monitor.flag_out_of_range([[np.inf]])
