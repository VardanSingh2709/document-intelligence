"""Unit tests for confidence-based routing (Phase 8's empirically-derived
thresholds). These encode the actual numbers from our validation analysis —
if someone changes a threshold without updating these tests deliberately,
the test failure forces a conscious decision, not a silent drift."""
from app.extraction.routing import FIELD_THRESHOLDS, should_accept


def test_date_always_escalates_even_at_maximum_confidence():
    assert should_accept("DATE", 1.0) is False


def test_total_accepts_above_its_threshold():
    assert should_accept("TOTAL", FIELD_THRESHOLDS["TOTAL"] + 0.01) is True


def test_total_rejects_below_its_threshold():
    assert should_accept("TOTAL", FIELD_THRESHOLDS["TOTAL"] - 0.01) is False


def test_none_confidence_is_never_accepted():
    # No prediction at all should never be treated as "accept".
    assert should_accept("TOTAL", None) is False


def test_company_has_a_stricter_threshold_than_total():
    # Documents the Phase 8 finding: COMPANY's confidence-correctness
    # relationship is weaker, so it was deliberately set more conservatively.
    assert FIELD_THRESHOLDS["COMPANY"] > FIELD_THRESHOLDS["TOTAL"]