# Phase 7: LayoutLMv3 vs. Rule-Based Baseline — Full Comparison

Evaluated on 346 test receipts (of 360 total; 14 excluded for having no
Task 3 entity ground truth, per the official SROIE competition scope — see
`data/README.md`). Both evaluations use ground-truth OCR text, isolating
extraction quality from OCR quality (see the planned end-to-end evaluation
for the full picture including our own OCR engine's errors).

| Field   | Metric            | Rules (Phase 4) | LayoutLMv3 (Phase 7) | Change  |
|---------|-------------------|------------------|------------------------|---------|
| Date    | F1                | 0.986            | 0.981                  | -0.005  |
| Total   | F1                | 0.554            | 0.940                  | +0.386  |
| Company | Fuzzy similarity  | 0.848            | 0.968                  | +0.120  |
| Address | Fuzzy similarity  | 0.718            | 0.982                  | +0.264  |

## Key finding

LayoutLMv3's improvement is concentrated exactly where Phase 4 found rules
to be structurally limited. `total` required extensive, brittle hand-tuned
heuristics (search windows, stop-words, keyword priority) to reach 0.554 F1,
documented across 5 debugging iterations in Phase 4's commit history.
LayoutLMv3 reaches 0.940 F1 on the same field without any of that manual
logic, learning the relevant positional/visual patterns directly from 553
training examples.

`date`, where rules were already near-perfect due to its simple, consistent
shape, shows no improvement from ML, a useful negative result: added model
complexity doesn't help a field that was never structurally hard. This
supports a practical design principle: reserve ML capacity for genuinely
ambiguous fields, and keep cheap, interpretable rules where they already work.

## Caveats

- Both evaluations use ground-truth OCR text, not our own OCR engine's
  output. A fully honest end-to-end number requires chaining our actual
  OCR (Phase 3) into this evaluation, which will show somewhat lower
  scores for both approaches, and is a natural next measurement.
- Company/address use fuzzy similarity, not F1; not directly comparable
  in scale to date/total's F1, though both show the same directional story.
- LayoutLMv3 was evaluated only on the fields it was trained on; this says
  nothing about its ability to generalize to new entity types without
  retraining.