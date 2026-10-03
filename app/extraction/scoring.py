"""Shared scoring utilities used by both the rule-based (Phase 4) and
LayoutLMv3 (Phase 7) evaluations, ensuring a fair, consistent comparison."""
import re
from difflib import SequenceMatcher


def normalize_money(value: str | None) -> float | None:
    """Compare totals as numbers, not strings. Strips currency symbols/codes so
    'RM4.00' and '4.00' are correctly treated as equal amounts."""
    if value is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def text_similarity(a: str | None, b: str | None) -> float:
    """Rough fuzzy match ratio (0-1) for freeform text fields where exact
    match is too strict."""
    if a is None or b is None:
        return 1.0 if a == b else 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def normalize_date(value: str | None) -> str | None:
    """Strip separators and leading zeros aren't touched — just normalize
    separators so '25/12/2018' and '25-12-2018' compare equal, matching the
    kind of minor formatting variation we accept elsewhere."""
    if value is None:
        return None
    return re.sub(r"[^0-9]", "", value)


def score_field(predictions: list, ground_truths: list) -> dict:
    """Compute precision, recall, F1 for one field across all receipts."""
    tp = fp = fn = 0
    for pred, truth in zip(predictions, ground_truths):
        if pred is None and truth is not None:
            fn += 1
        elif pred is not None and truth is None:
            fp += 1
        elif pred is not None and truth is not None:
            if pred == truth:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
