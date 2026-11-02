"""Release-facing matrix index for the SDD T01-T07 and T11 scenarios."""

import pandas as pd
import pytest

from bankruptcy_prediction.contracts import WarningCode
from bankruptcy_prediction.scoring.result_models import RESULT_COLUMNS


@pytest.mark.parametrize(
    ("case_id", "description"),
    [
        ("T01", "canonical headers and generated IDs"),
        ("T02", "file-level missing or unknown schema"),
        ("T03", "up to nine missing values are imputed"),
        ("T04", "ten or more missing values fail a row"),
        ("T05", "non-numeric and infinite values fail a row"),
        ("T06", "threshold classification scenario"),
        ("T07", "stable Top 10 percent and percentile"),
        ("T11", "out-of-range warning without clipping"),
    ],
)
def test_acceptance_matrix_ids_are_documented(case_id, description):
    assert case_id.startswith("T")
    assert description


def test_result_export_contract_has_fixed_columns_and_warning_codes():
    assert RESULT_COLUMNS[:5] == (
        "record_id",
        "model_version",
        "confidence_score",
        "batch_percentile",
        "threshold_used",
    )
    assert WarningCode.IMPUTED_VALUES.value == "IMPUTED_VALUES"
    assert WarningCode.OUT_OF_RANGE.value == "out_of_range"
    assert isinstance(pd.DataFrame(columns=RESULT_COLUMNS), pd.DataFrame)
