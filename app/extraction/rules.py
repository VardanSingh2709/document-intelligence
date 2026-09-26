"""Rule-based field extraction: regex and keyword heuristics for total and date.
No ML involved. This is our baseline every later model must beat."""
import re

# --- DATE ---
# Handles: 25/12/2018, 19/10/2018 (DD/MM/YYYY)
#          12-01-19             (DD-MM-YY)
# The (?:...) groups are non-capturing: we want the whole match, not sub-pieces.
DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
)

def extract_date(ocr_text: str) -> str | None:
    """Return the first date-shaped substring found, or None if none found."""
    match = DATE_PATTERN.search(ocr_text)
    return match.group(0) if match else None


# --- TOTAL ---
# A money-shaped number: optional currency symbol/code, digits, optional
# thousands separators, decimal point, exactly 2 decimal digits.
MONEY_PATTERN = re.compile(
    r"(?:RM|MYR|\$)?\s*\d{1,3}(?:,\d{3})*\.\d{2}"
)

# Ordered most-specific-first, though with "take the last match" logic below,
# keyword specificity matters less than it used to — kept for clarity and as
# a fallback signal.
TOTAL_KEYWORDS = ["total rounded", "grand total", "total sales", "amount due", "total"]
SEARCH_WINDOW = 6  # generous cap; the stop-word check below usually ends the search sooner
STOP_WORDS = ["cash", "change", "visa", "mastercard", "credit card", "debit"]


def extract_total(lines: list[str]) -> str | None:
    """Search for a total keyword, then return the LAST money-shaped value found
    before hitting a payment-method line (CASH, CHANGE, etc.) or SEARCH_WINDOW,
    whichever comes first. Receipts consistently place total/rounding lines
    immediately before the payment section, so that boundary is a reliable,
    structural stopping point rather than an arbitrary line count."""
    lowered = [line.lower() for line in lines]

    for keyword in TOTAL_KEYWORDS:
        for i, line in enumerate(lowered):
            if "qty" in line:
                continue
            if keyword in line and "subtotal" not in line:
                window_end = min(i + 1 + SEARCH_WINDOW, len(lines))
                for j in range(i + 1, window_end):
                    if any(stop in lowered[j] for stop in STOP_WORDS):
                        window_end = j  # stop before the payment line, don't include it
                        break

                matches = [MONEY_PATTERN.search(lines[j]) for j in range(i, window_end)]
                matches = [m for m in matches if m]
                if matches:
                    return _clean_money(matches[-1].group(0))
    return None


def _clean_money(raw: str) -> str:
    """Strip currency symbols/codes and thousands separators, keeping just digits and the decimal point."""
    cleaned = re.sub(r"[^\d.]", "", raw)
    return cleaned