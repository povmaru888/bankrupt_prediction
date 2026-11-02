import json
import logging

from bankruptcy_prediction.contracts import BatchSummary, RowStatus
from bankruptcy_prediction.logging_config import _JsonLineFormatter
from bankruptcy_prediction.scoring.result_models import ScoringResult, ScoringResultRow
from bankruptcy_prediction.service.audit_logger import record_score_request


def make_result():
    row = ScoringResultRow(
        record_id="secret-company",
        model_version="v1",
        row_status=RowStatus.FAILED,
        threshold_used=0.5,
        error_code="NON_NUMERIC_VALUE",
        error_message="secret financial feature",
    )
    return ScoringResult(BatchSummary(1, 0, 1, "v1", 0.5), (row,))


def test_record_score_request_uses_safe_aggregate_fields(caplog):
    logger = logging.getLogger("bankruptcy_prediction.audit")
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="bankruptcy_prediction.audit"):
            request_id = record_score_request(make_result(), elapsed_ms=12, request_id="req-1")
        assert request_id == "req-1"
        record = caplog.records[-1]
        payload = json.loads(handler.format(record))
        assert payload["total_rows"] == 1
        assert payload["error_code_counts"] == {"NON_NUMERIC_VALUE": 1}
        text = json.dumps(payload, ensure_ascii=False)
        assert "secret-company" not in text
        assert "financial feature" not in text
    finally:
        logger.removeHandler(handler)
