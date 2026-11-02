"""Deterministic 10,000-row scoring performance acceptance."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from bankruptcy_prediction.explanation.shap_service import (
    ExplanationBatch,
    ExplanationFactor,
    ExplanationRow,
)
from bankruptcy_prediction.scoring import service
from bankruptcy_prediction.scoring.service import score_batch


@dataclass(frozen=True)
class Active:
    version: str = "perf-v1"
    model_family: str = "logistic_regression"
    default_threshold: float = 0.5
    feature_schema: dict = None

    def __post_init__(self):
        object.__setattr__(
            self, "feature_schema", {"schema_version": 1, "feature_names": ("f1", "f2", "f3")}
        )

    model = None


class Model:
    def predict_proba(self, frame):
        score = np.clip(frame["f1"].to_numpy(float), 0, 1)
        return np.column_stack([1 - score, score])


def test_10000_rows_complete_scoring_and_serialization_under_ten_seconds(monkeypatch):
    rows = 10_000
    frame = pd.DataFrame({"f1": np.linspace(0, 1, rows), "f2": 0.0, "f3": 1.0})

    def explain_fast(model_family, model, X):
        del model_family, model
        factor = ExplanationFactor("f1", 0.5, 0.1)
        negative = ExplanationFactor("f2", 0.0, -0.1)
        row = ExplanationRow((factor,), (negative,))
        return ExplanationBatch(
            tuple(row for _ in range(len(X))), ("f1", "f2", "f3"), "logistic_regression"
        )

    monkeypatch.setattr(service, "explain", explain_fast)
    active = Active()
    object.__setattr__(active, "model", Model())
    started = perf_counter()
    result = score_batch(active, frame)
    encoded = result.to_json()
    elapsed = perf_counter() - started
    assert result.summary.succeeded_rows == rows
    assert len(encoded) > rows
    assert elapsed < 10.0
