# Phase 19: Final Evaluation Report

This report consolidates every measured result from Phases 4–9 into one
place. Every number below is cited to the evaluation file and run that
produced it — nothing here is a new calculation or an estimate.

## Important caveat on comparability

The columns below are **not** all measured on the same held-out set:

- **Rules** and **Hybrid** were evaluated on the **validation set** (73
  receipts, carved from training data in Phase 4, used for threshold
  selection in Phase 8).
- **LayoutLMv3 (clean text)** and **LayoutLMv3 (our OCR)** were evaluated
  on the **test set** (346 receipts, never used in training or threshold
  selection).

All four are genuinely out-of-sample relative to model training, so the
comparison is directionally meaningful (it correctly shows the relative
impact of OCR noise, and of the hybrid fallback). It is not a strict
apples-to-apples benchmark across identical data. A full re-run of the
hybrid system on the test set would close this gap; it was not performed
in this project due to Groq free-tier rate limits making repeated full
test-set runs costly in development time (see Phase 9 notes).

## Consolidated results

| Field   | Rules (val, clean text) | LayoutLMv3 (test, clean text) | LayoutLMv3 (test, our OCR) | Hybrid (val, our OCR) |
|---------|--------------------------|-------------------------------|------------------------------|--------------------------|
| Date    | 0.986 (F1)               | 0.981 (F1)                    | 0.549 (F1)                   | 0.950 (F1)               |
| Total   | 0.554 (F1)               | 0.940 (F1)                    | 0.905 (F1)                   | 0.950 (F1)               |
| Company | 0.848 (similarity)       | 0.968 (similarity)            | 0.885 (similarity)           | 0.832 (similarity)       |
| Address | 0.718 (similarity)       | 0.982 (similarity)            | 0.927 (similarity)           | 0.902 (similarity)       |

**Sources:**
- Rules column: `evaluation/phase4_baseline.md`
- LayoutLMv3 (clean text): `evaluation/phase7_layoutlm_results.md`
- LayoutLMv3 (our OCR): `evaluation/phase7_end_to_end_results.md`
- Hybrid (our OCR): `evaluation/phase9_hybrid_results.md`

Company/Address use fuzzy text similarity (difflib SequenceMatcher), not
F1; they are not on the same numeric scale as Date/Total and should not be
averaged together with them.

## The headline findings, in order of importance

**1. Real OCR substantially degrades DATE specifically, not fields uniformly.**
LayoutLMv3 drops from 0.981 to 0.549 F1 on DATE when moving from clean
ground-truth text to our own OCR output — a 0.432 collapse. Total, Company,
and Address degrade too, but far more gracefully (0.04–0.09). This was
root-caused in Phase 7/8: dates are short, dense tokens where a single
misread character (or a merged date+time token, see `known_issues.md` #3)
invalidates the whole value, and the model has no learned recovery pattern
for corrupted dates.

**2. The hybrid architecture recovers most of DATE's lost accuracy.**
Escalating 100% of DATE predictions to the LLM fallback (a decision made
from Phase 8's confidence analysis, not from seeing this outcome in
advance) raised DATE from 0.549 to 0.950 — within 0.031 of the clean-text
ceiling. This is the single strongest result in the project: a measured,
targeted fix for a measured, specific weakness.

**3. The hybrid fallback is not uniformly beneficial.** COMPANY and
ADDRESS both show small *regressions* under the hybrid system versus
LayoutLMv3-alone-with-OCR (Company: 0.885 → 0.832; Address: 0.927 → 0.902).
The likely cause (Phase 9 write-up): the LLM fallback sees only flattened
OCR text, losing the positional/visual signal LayoutLMv3 uses. This means
the LLM fallback is not a universal safety net — its benefit is field-
specific, and the project's own evidence argues for being more selective
about which fields warrant escalation.

**4. Rule-based extraction is not uniformly weak.** Rules nearly match
LayoutLMv3 on DATE (0.986 vs. 0.981, on clean text) because dates have a
simple, regular shape regex can exploit. Rules are weakest specifically on
TOTAL (0.554), where keyword-adjacent values cluster ambiguously (recall
the 5 debugging iterations documented in Phase 4's commit history). This
means the honest conclusion is "ML earns its complexity where structure is
genuinely ambiguous," not "ML beats rules everywhere."

## System performance (latency)

| Stage | Observed timing | Source |
|---|---|---|
| OCR (PaddleOCR, CPU) | ~2.6–3.3s/receipt | Phase 3 benchmark |
| LayoutLMv3 inference (GPU) | ~0.46s/receipt average | Phase 9 hybrid eval |
| LLM fallback call (when escalated) | ~4.13s/receipt average (escalated fields only) | Phase 9 hybrid eval |
| LayoutLMv3 inference (CPU, in Docker) | not formally benchmarked; qualitatively several times slower per manual Phase 15 test | Phase 15 notes |

DATE's 100% escalation rate means **every** processed receipt pays the LLM
fallback's latency cost at least once — a real, accepted tradeoff given the
accuracy recovery it buys (Finding 2 above).

## What this report does not cover

- Cost in dollar terms: the LLM fallback used Groq's free tier throughout
  development; no paid-tier cost analysis was performed (see Phase 9's
  production-deployment caveat).
- A test-set run of the full hybrid system (see caveat above).
- Formal failure categorization — covered separately in Phase 20.