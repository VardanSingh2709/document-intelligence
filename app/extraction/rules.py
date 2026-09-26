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


# --- COMPANY & ADDRESS ---
# Purely positional heuristics: unlike date/total, company names and addresses
# have no consistent textual shape, so we rely on where they typically sit on
# a receipt (top of the document) rather than what they contain.
ADDRESS_STOP_MARKERS = ["tel", "fax", "invoice", "receipt", "gst", "www", "email"]


KNOWN_HEADER_ARTIFACT = "tan woon yann"  # a non-company line appearing first on nearly every SROIE receipt


def extract_company(lines: list[str]) -> str | None:
    """Assume the company name is the first non-empty line, skipping a known
    dataset artifact ("TAN WOON YANN") that appears as line 0 on most receipts
    regardless of actual vendor (discovered via evaluation, see phase4_baseline.md)."""
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.lower() != KNOWN_HEADER_ARTIFACT:
            return stripped
    return None


def extract_address(lines: list[str]) -> str | None:
    """Assume the address follows immediately after the company line (line 0),
    continuing until a stop marker (phone, invoice label, etc.) appears."""
    address_lines = []
    for line in lines[1:5]:  # look at up to 4 lines after the company line
        lowered = line.lower()
        if any(marker in lowered for marker in ADDRESS_STOP_MARKERS):
            break
        if DATE_PATTERN.search(line):
            break
        address_lines.append(line.strip())
    return " ".join(address_lines) if address_lines else None