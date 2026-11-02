"""UTF-8 CSV export with record ID spreadsheet-formula injection protection."""

from __future__ import annotations

from io import StringIO

from bankruptcy_prediction.scoring.result_models import ScoringResult

FORMULA_PREFIXES = ("=", "+", "-", "@")


def export_results_csv(result: ScoringResult) -> bytes:
    """Serialize scoring results in fixed column order without changing source DTOs."""
    frame = result.to_dataframe()
    frame["record_id"] = frame["record_id"].map(
        lambda value: f"'{value}" if value.startswith(FORMULA_PREFIXES) else value
    )
    buffer = StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")
