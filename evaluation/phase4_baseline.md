# Phase 4: Rule-Based Baseline Results

Evaluated on 73 held-out validation receipts (company-grouped split, see
`data/raw/sroie/train_val_split.json`), against ground-truth OCR text
(not our own OCR output — this isolates rule quality from OCR quality;
see Phase 7 for the full end-to-end number).

| Field   | Metric             | Score |
|---------|--------------------|-------|
| Date    | F1                 | 0.986 |
| Total   | F1                 | 0.554 |
| Company | Fuzzy similarity*  | 0.848 |
| Address | Fuzzy similarity*  | 0.718 |

*Company/address use character-level fuzzy similarity (difflib SequenceMatcher),
not F1, since exact string match is too strict for freeform text. These numbers
are not directly comparable to date/total's F1 scores.

## Approach

- **Date**: single regex pattern matching `DD/MM/YYYY` and `DD-MM-YY` style dates.
- **Total**: keyword search followed by the *last* money-shaped value within a
  window of subsequent lines, stopping at payment-related lines. Derived
  empirically after multiple rounds of debugging against real receipts (see
  commit history) — receipts consistently show a raw total, a rounding
  adjustment, then the final rounded total, in that order.
- **Company**: the first non-empty OCR line, skipping a known dataset artifact
  ("TAN WOON YANN") that appears as line 0 on nearly every SROIE receipt
  regardless of actual vendor — discovered via evaluation (this heuristic
  scored 0.000 before the fix).
- **Address**: the 1-4 lines following the company line, stopping at markers
  like "TEL", "FAX", "INVOICE", or a detected date.

## Key finding: structure predicts difficulty better than "freeform vs. fixed-shape"

Our initial hypothesis was that freeform fields (company, address) would
underperform structured fields (date, total) with pure rules. The results
partially contradict this: company name, though freeform, scored highly
(0.848) because it reliably sits at a fixed *position* (top of receipt),
once a dataset-specific artifact was accounted for. Total, despite being
tightly patterned, scored lowest (0.554) because its value clusters with
several other money-like numbers (tax, rounding, cash tendered) in
inconsistent OCR line arrangements. Position and shape both matter, and
neither alone predicts how tractable a field is for rule-based extraction.

## Known failure patterns for TOTAL (see mismatches above)

1. Search window sometimes lands on a `0.00` rounding-adjustment line
   instead of the true total.
2. Some receipts phrase the total in ways not covered by our keyword list,
   yielding no prediction at all.
3. A few receipts show large, unexplained mismatches, likely a keyword
   match against an unrelated part of the receipt.

## What this means for Phase 6

This baseline is the floor Phase 6's LayoutLMv3 model must beat to justify
its complexity. Given these results, `total` is the clearest candidate for
ML to add real value; `date` may see limited improvement since rules
already perform very well; `company` and `address` sit in between and are
genuinely open questions worth measuring rather than assuming.