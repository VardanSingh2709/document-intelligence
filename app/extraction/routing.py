"""Confidence-based routing: decide whether a field's prediction should be
accepted automatically or flagged for further handling (LLM fallback /
human review, built in Phase 9), based on thresholds derived empirically
from the validation set (see evaluation/phase8_routing_thresholds.md)."""

# Derived from validation-set threshold sweep (Phase 8). TOTAL and ADDRESS
# threshold chosen where accuracy plateaus without meaningfully sacrificing
# accept-rate. COMPANY set more conservatively given its weaker confidence-
# correctness relationship even at high confidence.
FIELD_THRESHOLDS = {
    "TOTAL": 0.90,
    "ADDRESS": 0.90,
    "COMPANY": 0.95,
}

# DATE is excluded from FIELD_THRESHOLDS deliberately: validation analysis
# showed confidence does not meaningfully separate correct from incorrect
# date predictions (mean conf. 0.956 correct vs. 0.932 incorrect — a gap of
# just 0.024, versus 0.14-0.18 for other fields). Thresholding this field
# would give false reassurance. Every date prediction is routed onward
# regardless of confidence.
ALWAYS_ESCALATE_FIELDS = {"DATE"}


def should_accept(field: str, confidence: float | None) -> bool:
    """Return True if this field's prediction should be accepted automatically,
    False if it should be routed onward (Phase 9's fallback/review)."""
    if field in ALWAYS_ESCALATE_FIELDS:
        return False
    if confidence is None:
        return False  # no prediction at all — nothing to accept
    threshold = FIELD_THRESHOLDS.get(field, 0.90)  # safe default for any future field
    return confidence >= threshold