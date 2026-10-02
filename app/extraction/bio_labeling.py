"""Match entity ground-truth strings to word tokens and produce BIO labels
for LayoutLMv3 training."""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.extraction.tokenize_boxes import WordToken

LABEL_FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]


def normalize(text: str) -> str:
    """Lowercase and strip everything except letters/digits, so minor
    punctuation/spacing differences don't prevent a match."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class LabeledToken:
    text: str
    box: list[int]
    label: str  # e.g. "O", "B-TOTAL", "I-TOTAL"


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def find_best_span(tokens: list[WordToken], entity_text: str) -> tuple[int, int] | None:
    """Find the contiguous run of tokens [start, end) whose concatenated,
    normalized text best matches the normalized entity_text. Returns None if
    no reasonably good match exists (threshold-based, not forced).
    """
    target = normalize(entity_text)
    if not target:
        return None

    best_score = 0.0
    best_span = None

    max_span_len = min(len(tokens), 25)

    for start in range(len(tokens)):
        concatenated = ""
        for length in range(1, max_span_len + 1):
            end = start + length
            if end > len(tokens):
                break
            concatenated += normalize(tokens[end - 1].text)

            if len(concatenated) > len(target) * 1.5:
                break

            score = _similarity(concatenated, target)
            if score > best_score:
                best_score = score
                best_span = (start, end)

    MATCH_THRESHOLD = 0.85
    if best_score >= MATCH_THRESHOLD:
        return best_span
    return None


def apply_bio_labels(tokens: list[WordToken], entities: dict[str, str]) -> tuple[list[LabeledToken], dict[str, bool]]:
    """Label every token as O, or B-<FIELD>/I-<FIELD> for matched entity spans.
    Returns the labeled tokens plus a dict recording which fields were
    successfully matched (for statistics)."""
    labels = ["O"] * len(tokens)
    matched = {}

    for field in LABEL_FIELDS:
        entity_text = entities.get(field.lower()) or ""   # handles both missing key AND explicit null
        span = find_best_span(tokens, entity_text)
        matched[field] = span is not None

        if span:
            start, end = span
            for i in range(start, end):
                if labels[i] == "O":
                    labels[i] = f"{'B' if i == start else 'I'}-{field}"

    labeled_tokens = [
        LabeledToken(text=tok.text, box=tok.box, label=label)
        for tok, label in zip(tokens, labels)
    ]
    return labeled_tokens, matched