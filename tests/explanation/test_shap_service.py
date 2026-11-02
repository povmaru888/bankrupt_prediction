"""Tests for SHAP explainer selection and factor projections."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from bankruptcy_prediction.explanation import shap_service
from bankruptcy_prediction.explanation.shap_service import explain


class _Selector:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, ["a", "b", "c", "d", "e", "f"]]


class _Scaler:
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=float)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(["a", "b", "c", "d", "e", "f"])


class _Logistic:
    coef_ = np.ones((1, 6))
    intercept_ = np.array([0.0])


class _Tree:
    pass


class _LinearExplainer:
    calls = 0

    def __init__(self, model: object, background: np.ndarray) -> None:
        self.__class__.calls += 1
        assert background.shape == (1, 6)
        self.values = np.array([[5, 4, 3, -1, -2, -3]], dtype=float)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.values, (len(X), 1))


class _TreeExplainer:
    calls = 0

    def __init__(self, model: object) -> None:
        self.__class__.calls += 1

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[1, 2, 3, -3, -2, -1]], dtype=float), (len(X), 1))


def _pipeline(classifier: object) -> Pipeline:
    return Pipeline([("selector", _Selector()), ("scaler", _Scaler()), ("classifier", classifier)])


@pytest.fixture(autouse=True)
def _mock_shap(monkeypatch: pytest.MonkeyPatch) -> None:
    _LinearExplainer.calls = 0
    _TreeExplainer.calls = 0
    monkeypatch.setattr(
        shap_service,
        "_shap_module",
        lambda: SimpleNamespace(LinearExplainer=_LinearExplainer, TreeExplainer=_TreeExplainer),
    )


def test_logistic_uses_linear_shap_and_sorts_both_directions() -> None:
    frame = pd.DataFrame([[10, 20, 30, 40, 50, 60]], columns=list("abcdef"))

    result = explain("logistic_regression", _pipeline(_Logistic()), frame)
    row = result.rows[0]

    assert _LinearExplainer.calls == 1
    assert _TreeExplainer.calls == 0
    assert [(factor.feature_name, factor.contribution) for factor in row.positive_factors] == [
        ("a", 5),
        ("b", 4),
        ("c", 3),
    ]
    assert [(factor.feature_name, factor.contribution) for factor in row.negative_factors] == [
        ("f", -3),
        ("e", -2),
        ("d", -1),
    ]
    assert row.positive_factors[0].feature_value == 10


def test_xgboost_uses_tree_shap_and_exports_top_three() -> None:
    frame = pd.DataFrame([[10, 20, 30, 40, 50, 60]], columns=list("abcdef"))

    result = explain("xgboost", _pipeline(_Tree()), frame)

    assert _TreeExplainer.calls == 1
    assert _LinearExplainer.calls == 0
    assert len(result.rows[0].positive_factors) == 3
    assert len(result.rows[0].negative_factors) == 3
    assert len(result.project(3).rows[0].positive_factors) == 3
    assert len(result.project(1).rows[0].negative_factors) == 1


def test_calibrated_ensemble_averages_base_explanations() -> None:
    base = _pipeline(_Logistic())
    calibrated = SimpleNamespace(
        calibrated_classifiers_=[SimpleNamespace(estimator=base), SimpleNamespace(estimator=base)]
    )
    frame = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=list("abcdef"))

    result = explain("logistic_regression", calibrated, frame)

    assert _LinearExplainer.calls == 2
    assert result.rows[0].positive_factors[0].contribution == 5


@pytest.mark.parametrize(
    "family,frame",
    [
        ("other", pd.DataFrame([[1]], columns=["a"])),
        ("xgboost", pd.DataFrame()),
        ("xgboost", pd.DataFrame([[1, 2]], columns=["a", "a"])),
        ("xgboost", pd.DataFrame([["not numeric"]], columns=["a"])),
    ],
)
def test_rejects_invalid_family_or_input(family: str, frame: pd.DataFrame) -> None:
    with pytest.raises((TypeError, ValueError)):
        explain(family, _pipeline(_Tree()), frame)
