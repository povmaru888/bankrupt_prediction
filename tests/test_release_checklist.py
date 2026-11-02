from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_documents_trace_f01_to_f10_and_state_blocker_honestly():
    checklist = (ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    report = (ROOT / "reports" / "acceptance_report.md").read_text(encoding="utf-8")
    for requirement in ("F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10"):
        assert requirement in checklist and requirement in report
    assert "Not Ready for Release" in report
    assert "active model" in report
