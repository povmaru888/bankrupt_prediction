"""Tests for fold-fitted median imputation and warning propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from bankruptcy_prediction.contracts import WarningCode
from bankruptcy_prediction.features.imputation import build_imputer


def test_median_is_fitted_from_training_values_only_and_not_refit_on_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [2.0, 4.0, 6.0]})
    validation = pd.DataFrame({"a": [1000.0], "b": [2000.0]})
    test = pd.DataFrame({"a": [-1000.0], "b": [-2000.0]})
    imputer = build_imputer()
    original_fit = imputer.fit
    fit_inputs: list[np.ndarray] = []

    def fit_spy(X, y=None):
        fit_inputs.append(np.asarray(X, dtype=float).copy())
        return original_fit(X, y)

    monkeypatch.setattr(imputer, "fit", fit_spy)
    imputer.fit(train)
    validation_out = imputer.transform(validation)
    test_out = imputer.transform(test)

    assert imputer.statistics_.tolist() == [3.0, 4.0]
    assert validation_out.tolist() == [[1000.0, 2000.0]]
    assert test_out.tolist() == [[-1000.0, -2000.0]]
    assert len(fit_inputs) == 1
    np.testing.assert_array_equal(fit_inputs[0], train.to_numpy())


def test_imputes_rows_with_up_to_nine_missing_values_and_sets_warning(
    nine_missing_batch: pd.DataFrame, feature_names: tuple[str, ...]
) -> None:
    train = pd.DataFrame(
        np.vstack([np.zeros(len(feature_names)), np.ones(len(feature_names))]),
        columns=feature_names,
    )
    imputer = build_imputer().fit(train)
    sample = nine_missing_batch.drop(columns="record_id")

    transformed, warnings = imputer.transform_with_warnings(sample)

    assert transformed.shape == (2, 95)
    assert not np.isnan(transformed).any()
    assert warnings == ((WarningCode.IMPUTED_VALUES.value,), ())
    assert nine_missing_batch.iloc[0].isna().sum() == 9


def test_rejects_transform_before_fit_and_all_missing_training_feature() -> None:
    with pytest.raises(NotFittedError):
        build_imputer().transform([[1.0]])

    with pytest.raises(ValueError, match="all-missing training features"):
        build_imputer().fit([[np.nan, 1.0], [np.nan, 2.0]])


def test_imputer_is_sklearn_cloneable_and_checks_feature_order() -> None:
    train = pd.DataFrame({"first": [1.0, 3.0], "second": [10.0, 30.0]})
    imputer = clone(build_imputer()).fit(train)

    with pytest.raises(ValueError, match="feature names or order"):
        imputer.transform(train.loc[:, ["second", "first"]])
