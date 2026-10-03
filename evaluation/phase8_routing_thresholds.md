# Phase 8: Confidence-Based Routing Thresholds

Derived from confidence-vs-correctness analysis on 73 validation receipts,
evaluated through our own OCR (not ground-truth text), since routing
decisions must reflect real-world confidence behavior.

## Confidence separation by field

| Field   | Mean conf. (correct) | Mean conf. (incorrect) | Gap   |
|---------|----------------------|--------------------------|-------|
| Total   | 0.992                | 0.852                    | 0.140 |
| Address | 0.978                | 0.828                    | 0.150 |
| Company | 0.984                | 0.804                    | 0.180 |
| Date    | 0.956                | 0.932                    | 0.024 |

## Chosen thresholds

| Field   | Threshold | Accept rate | Accuracy of accepted |
|---------|-----------|-------------|------------------------|
| Total   | 0.90      | 94%         | 92.6%                  |
| Address | 0.90      | 84%         | 83.6%                  |
| Company | 0.95      | 53%         | 81.6%                  |
| Date    | N/A — always escalated | 0% | N/A |

## Key finding: confidence is not reliable for the DATE field

Despite high absolute confidence values, DATE shows almost no separation
between correct and incorrect predictions (a 0.024 gap, vs. 0.14-0.18 for
other fields). This means the model's self-reported certainty cannot be
trusted to flag bad date extractions — it is often just as "confident" when
wrong as when right. This is consistent with Phase 7's finding that OCR
errors disproportionately corrupt dates (short, dense tokens where one
misread character invalidates the whole value) in a way the model wasn't
trained to recognize as suspicious.

Rather than set an arbitrary, falsely-reassuring threshold for this field,
every date prediction is routed to the fallback stage (Phase 9) regardless
of confidence.

## A secondary finding: COMPANY's confidence is less trustworthy than TOTAL/ADDRESS

Even at a 0.97 threshold, COMPANY predictions are only correct 82.9% of the
time, versus TOTAL's 96.9% at the same threshold. This likely reflects
COMPANY's open-ended text space (many plausible business names) compared to
TOTAL's narrower, more structurally anchored value space. COMPANY's
threshold (0.95) was set conservatively in response to this weaker signal.