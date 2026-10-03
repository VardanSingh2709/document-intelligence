# Phase 7: Full End-to-End Evaluation (Our OCR + LayoutLMv3)

Evaluated on 346 test receipts, using our own PaddleOCR engine (Phase 3)
feeding into the trained LayoutLMv3 model (Phase 6), compared against both
the rule-based baseline and LayoutLMv3 on ground-truth OCR text.

| Field   | Rules (ground-truth text) | LayoutLMv3 (ground-truth text) | LayoutLMv3 (our OCR, end-to-end) |
|---------|---------------------------|----------------------------------|-------------------------------------|
| Date    | 0.986 (F1)                | 0.981 (F1)                       | 0.549 (F1)                          |
| Total   | 0.554 (F1)                | 0.940 (F1)                       | 0.905 (F1)                          |
| Company | 0.848 (similarity)        | 0.968 (similarity)               | 0.885 (similarity)                  |
| Address | 0.718 (similarity)        | 0.982 (similarity)               | 0.927 (similarity)                  |

## Key finding: OCR errors affect fields unevenly, and date is the most fragile

Total, company, and address all degrade gracefully under real OCR (roughly
4-10 points), and remain strong improvements over the rule-based baseline.
Date degrades sharply (0.981 -> 0.549), losing most of its advantage.

This is explained by date's token structure: a short, dense sequence
(e.g., "25/12/2018") where a single misread character corrupts the entire
value. LayoutLMv3 was trained only on clean, ground-truth-formatted date
tokens and has no learned recovery pattern for a garbled token — unlike
longer fields (address, company) where a single wrong character in a
30-character span barely affects overall similarity, or total, where the
model can anchor on surrounding context (keywords, position) even if one
digit is misread.

## Practical implication

This finding directly motivates Phase 8's confidence-based routing: date
extractions with low model confidence should route to an LLM fallback or
flag for human review more aggressively than other fields, since this
analysis shows date is disproportionately vulnerable to upstream OCR noise
in a way the other three fields are not.