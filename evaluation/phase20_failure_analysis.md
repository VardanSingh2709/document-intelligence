# Phase 20: Failure Analysis

This report categorizes every confirmed failure mode found during
development and evaluation (Phases 3–12), against the standard categories
used for document-AI system review. Each entry cites its source. Two
categories have no confirmed example in this project; that gap is stated
explicitly rather than filled artificially.

## 1. OCR failure

**What happened:** PaddleOCR merges adjacent tokens with no space between
them when spacing is tight in the source image. Two concrete, measured
consequences:
- A receipt's date and its adjacent timestamp were merged into one
  unparseable token (`10/03/2018 17:24:07` → `10/03/20181724:07`),
  destroying the date's recognizable structure for every downstream
  component (`known_issues.md` #3).
- General word-merging (e.g., `"NO.555,57&59JALANSAGU18"` instead of
  `"NO.53 55,57 & 59, JALAN SAGU 18"`) measured at ~13% character error
  rate overall (Phase 3 benchmark).

**Why it happened:** PaddleOCR's recognition model has no explicit
space-prediction signal for tightly-kerned or low-resolution text; this is
a documented, general weakness of the engine, not specific to our setup.

**How we'd improve it:** evaluate an OCR engine with stronger built-in
word-segmentation (we only benchmarked PaddleOCR and Tesseract — Phase 3);
alternatively, add a post-OCR heuristic that re-inserts likely spaces
around date/time patterns and currency amounts before tokenization, since
these have predictable shapes a rule could target even without full OCR
improvement.

## 2. Layout ambiguity / unexpected document format

**What happened:** the same receipt's date had no nearby label (no "DATE:"
prefix), and sat embedded inline within an order/register code line
(`"ORD #07 -REG #19- 10/03/2018 17:24:07"`). Both LayoutLMv3 and the LLM
fallback failed to recognize it, even given the (corrupted) OCR text —
and critically, the LLM fallback *also* failed on the clean ground-truth
version of this same unlabeled layout in early testing, before we
discovered the real blocker was the OCR merge (`known_issues.md` #3,
Phase 10 CLI test).

**Why it happened:** our training data (SROIE) predominantly shows dates
in labeled, isolated positions (`"DATE: 25/12/2018"` on its own line).
The model has limited exposure to the "unlabeled, embedded in a code
cluster" layout this receipt uses.

**How we'd improve it:** augment training data with unlabeled/embedded
date examples, or add a regex-based pre-scan as a secondary signal
(reintroducing a limited, targeted version of the Phase 4 rule-based
approach specifically as a fallback cue for the model, not a replacement).

## 3. Wrong field association / model confusion

**What happened:** TOTAL is predicted as `"3.50 8.50"` — two concatenated,
unrelated values — on receipts with multiple money-like numbers near
"TOTAL" labels (an item price and the real total appearing under two
different labels). Confirmed reproducible at high confidence (0.966),
meaning it is accepted automatically without escalation (`known_issues.md`
#1).

**Why it happened:** our BIO span-reconstruction logic (Phase 6/7) joins
all consecutive same-label tokens into one string. If the model predicts
`B-TOTAL`/`I-TOTAL` for two separate, non-adjacent regions (because both
look locally similar to "a total"), reconstruction silently merges them
into one malformed value rather than detecting the inconsistency.

**Why this is the most operationally concerning finding in the project:**
unlike the DATE failures (which correctly surface as "no value" and
escalate), this failure produces a *plausible-looking but wrong* value at
*high confidence*. It would pass Phase 8's routing threshold and ship
silently to a user. This is a more dangerous failure mode than an honest
null.

**How we'd improve it:** add a validation step after span reconstruction —
reject or down-weight a field's confidence if its token span is
non-contiguous, or if a money field's reconstructed text contains more
than one decimal-formatted number. This is a cheap, high-value fix we did
not implement in this project due to time; it is the top candidate for
future work.

## 4. Confidence miscalibration

**What happened:** Phase 8's validation analysis found DATE's confidence
is nearly identical whether the prediction is correct or incorrect (mean
0.956 vs. 0.932 — a 0.024 gap, versus 0.14–0.18 for every other field).
The model is, in effect, equally "sure" when right and when wrong on this
field.

**Why it happened:** likely because DATE's training examples (clean,
well-formatted ground truth) give the model no exposure to what a
*corrupted* date token looks like, so it has no learned signal
distinguishing "a clean date I recognize" from "a garbled token I'm
guessing is a date."

**How we'd improve it:** this is the one failure mode we *did* act on in
this project — DATE was set to always escalate regardless of confidence
(Phase 8), which Phase 9 confirmed recovered the field's accuracy almost
entirely (0.549 → 0.950 F1, Phase 19). The broader lesson: per-field
confidence calibration should be measured before trusting any
threshold, not assumed uniform across fields — a finding generalizable
beyond this project.

## 5. LLM hallucination — not observed

Across all testing, the LLM fallback was never observed to invent a
plausible-but-wrong value. When it could not confidently extract a field
(e.g., `known_issues.md` #2, ADDRESS given heavily OCR-corrupted text), it
consistently returned null rather than guessing. This is a genuinely
positive finding worth stating plainly rather than omitting: the fallback
model's prompt (explicitly instructing it to report uncertainty rather
than guess — Phase 9) appears effective at preventing this known LLM
failure mode, at least within this project's test volume (low hundreds of
calls). A larger-scale test would be needed to rule out rare hallucination
with statistical confidence; that was out of scope here.

## 6. Poor image quality — not specifically investigated

SROIE's images are scanned/photographed receipts of generally consistent
quality; this project did not specifically stress-test extreme cases
(heavy blur, glare, crumpled or faded thermal paper, extreme skew). It is
likely that such cases would compound the OCR failures already documented
above, but this was not measured and should not be assumed without
evidence. A natural extension would be synthetically degrading a sample
of test images (blur, rotation, contrast reduction) and re-running the
Phase 7 end-to-end evaluation to quantify robustness.

## Summary: what we'd fix first, given limited time

1. **Span-validation for TOTAL** (Section 3) — highest priority, since it
   silently produces wrong, high-confidence output rather than an honest
   failure.
2. **Per-field confidence calibration as a standard practice** (Section 4)
   — already applied to DATE; ADDRESS and COMPANY's weaker
   confidence-correctness relationship (Phase 8) suggests they deserve
   the same scrutiny, not just a static threshold.
3. **OCR engine evaluation beyond PaddleOCR/Tesseract** (Section 1) — the
   single highest-leverage upstream fix, since OCR errors are the root
   cause behind Sections 1, 2, and indirectly 4.