# Phase 4: Rule-Based Baseline Results

Evaluated on 73 held-out validation receipts (company-grouped split, see
`data/raw/sroie/train_val_split.json`), against ground-truth OCR text
(not our own OCR output — this isolates rule quality from OCR quality;
see Phase 7 for the full end-to-end number).

| Field | Precision | Recall | F1    |
|-------|-----------|--------|-------|
| Date  | 0.986     | 0.986  | 0.986 |
| Total | 0.609     | 0.509  | 0.554 |

## Approach

- **Date**: single regex pattern matching `DD/MM/YYYY` and `DD-MM-YY` style
  dates (the two formats observed in the training data).
- **Total**: keyword search (`total`, `grand total`, `total rounded`, etc.)
  followed by the *last* money-shaped value within a small window of
  subsequent lines, stopping early at payment-related lines (`cash`,
  `change`, etc.). This "last value" heuristic was derived empirically:
  receipts consistently show a raw total, a rounding adjustment, then the
  final rounded total, in that order.

## Why the gap between fields

Dates have one consistent shape (digits and separators) and appear once,
unambiguously, per receipt. Totals are structurally messy: multiple
money-like values cluster near the keyword (raw total, tax, rounding
adjustment, cash tendered, change), OCR line-splitting is inconsistent
(sometimes label+value share a line, sometimes they're split across
several single-token lines), and keyword phrasing varies
("TOTAL", "TOTAL ROUNDED", "TOTAL AMT", with OCR typos like "ROUND D
TOTAL"). A fixed-window, keyword-anchored regex approach has a real,
demonstrated ceiling against this variability.

## Known failure patterns (see mismatches for detail)

1. Search window sometimes lands on a `0.00` rounding-adjustment line
   instead of the true total.
2. Some receipts phrase the total in ways not covered by our keyword list,
   yielding no prediction at all.
3. A few receipts show large, unexplained mismatches, likely a keyword
   match against an unrelated part of the receipt.

This baseline establishes the floor for Phase 6: a layout-aware ML model
(LayoutLMv3) should meaningfully exceed 0.554 F1 on `total` to justify its
added complexity.