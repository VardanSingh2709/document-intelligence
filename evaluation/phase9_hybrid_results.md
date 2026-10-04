# Phase 9: Hybrid Pipeline (LayoutLMv3 + LLM Fallback) Results

Evaluated on 73 validation receipts, through our own OCR engine, using
Phase 8's empirically-derived confidence thresholds to decide escalation.

| Field   | LayoutLMv3 only (our OCR) | Hybrid (our OCR) | Change  |
|---------|-----------------------------|---------------------|---------|
| Date    | 0.549 (F1)                  | 0.950 (F1)           | +0.401  |
| Total   | 0.905 (F1)                  | 0.950 (F1)           | +0.045  |
| Company | 0.885 (similarity)          | 0.832 (similarity)   | -0.053  |
| Address | 0.927 (similarity)          | 0.902 (similarity)   | -0.025  |

## Escalation rates and cost

| Field   | Escalation rate | 
|---------|------------------|
| Date    | 100.0% (always, by design) |
| Company | 47.9%            |
| Address | 16.4%            |
| Total   | 6.8%             |

Average latency: 0.46s/receipt for the base model; +4.13s/receipt when a
field escalates to the LLM fallback (roughly 9x slower per call than the
base model, likely due to the fallback model's internal reasoning tokens —
observed during debugging to add 200-350+ hidden tokens before the visible
answer).

## Key finding: the fallback strongly helps DATE, is neutral-to-negative for others

The architectural decision to always escalate DATE (made in Phase 8, based
purely on confidence-correctness analysis, before this outcome was known)
is strongly validated: DATE's end-to-end F1 recovered from 0.549 to 0.950,
nearly matching ground-truth-text performance (0.981).

TOTAL improved modestly (+0.045). COMPANY and ADDRESS both saw small
regressions when escalated. A likely explanation: the LLM fallback sees
only flattened OCR text, losing the positional and visual information
LayoutLMv3 uses (see Phase 5) that may be particularly valuable for
disambiguating company names and addresses specifically.

## Honest implication for Phase 8's thresholds

This result suggests COMPANY's 0.95 threshold (and 47.9% escalation rate)
may not be justified by the evidence — the fallback doesn't clearly help
this field, and costs significant latency on nearly half of all receipts.
A natural refinement (not yet implemented) would be to raise COMPANY's
bar for escalation, or explore giving the LLM fallback more context (e.g.,
the receipt image) rather than text alone.