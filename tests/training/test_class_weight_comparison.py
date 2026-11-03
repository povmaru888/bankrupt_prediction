"""Tests for the controlled XGBoost class-weight ablation."""

import pytest

from bankruptcy_prediction.training.class_weight_comparison import (
    controlled_unweighted_parameters,
)


def test_controlled_unweighted_parameters_changes_only_weight() -> None:
    source = {
        "classifier__max_depth": 9,
        "classifier__learning_rate": 0.02,
        "classifier__scale_pos_weight": "balanced",
    }
    result = controlled_unweighted_parameters(source)
    assert result == {
        "classifier__max_depth": 9,
        "classifier__learning_rate": 0.02,
        "classifier__scale_pos_weight": 1.0,
    }
    assert source["classifier__scale_pos_weight"] == "balanced"


def test_controlled_unweighted_parameters_rejects_nonbalanced_source() -> None:
    with pytest.raises(ValueError, match="fold-balanced"):
        controlled_unweighted_parameters({"classifier__scale_pos_weight": 1.0})
