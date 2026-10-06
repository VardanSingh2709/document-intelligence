# Known Issues (tracked for Phase 20 failure analysis)

## 1. TOTAL field sometimes concatenates two values
**Example:** receipt X51005711453 (and its uploaded copies) consistently
predicts TOTAL as "3.50 8.50" instead of "8.50", at high confidence (0.966),
so it's accepted automatically without escalation.
**Likely cause:** BIO span reconstruction joins two separate, non-contiguous
B-TOTAL/I-TOTAL predictions into one string. The receipt has multiple
money-like values near "TOTAL" labels (an item price of 3.50, and the real
total of 8.50 appearing twice under different labels).
**Status:** confirmed reproducible, not yet fixed.

## 2. LLM fallback can fail on heavily OCR-corrupted text
**Example:** same receipt, ADDRESS field. Our OCR corrupted "Level" to
"Leve1" and "Petaling" to "Peta1ing" (1/l character confusion). The LLM
fallback, given only this corrupted flattened text, failed to confidently
extract a value, and correctly returned None rather than guessing.
**Likely cause:** the fallback has no visual/positional context (see Phase 9
write-up), unlike LayoutLMv3, so severely corrupted text gives it little to
work with.
**Status:** confirmed, behaves as a correct "I don't know" rather than a
wrong answer — arguably acceptable, but worth documenting as a real limit.

## 3. OCR can merge a date and adjacent time into one unparseable token
**Example:** receipt X51005711453. Ground truth has the date and time as
separate tokens ("10/03/2018 17:24:07"). Our OCR (PaddleOCR) merges them
into one token: "10/03/20181724:07", destroying the date's recognizable
structure. Neither LayoutLMv3 nor the LLM fallback (given this corrupted
text) can recover the correct date, both correctly report "no value" rather
than guessing. Confirmed via direct testing: the LLM fallback succeeds
(100% consistently across 5 trials) when given clean ground-truth text for
this receipt, and fails only when given our actual OCR's merged token —
isolating the failure cleanly to OCR, not to extraction logic.
**Status:** confirmed, root-caused. A genuine limitation of word-level OCR
on dense, unlabeled timestamp/date clusters — connects directly to the
Phase 7/8 finding that DATE is disproportionately OCR-fragile.

## 4. (Fixed) model_fallback_failed branch mislabeled its source
A copy/paste error in hybrid_pipeline.py's else-branch (both model and LLM
failed) incorrectly set source="llm_fallback" instead of
"model_fallback_failed", and discarded the model's last-attempt value in
favor of a guaranteed-null llm_value. This caused the frontend's "needs
review, editable" UI path to never trigger, even when the backend
correctly identified the field as needing review (confirmed via the
review-queue database, which was unaffected — only the API response label
was wrong). Found via direct comparison of isolated extract_field() calls
(worked correctly) against the full pipeline (showed wrong labels), and
fixed by correcting the else-branch's dict construction.

## 5. (Fixed) .env quotes broke Docker's --env-file parsing
Local development used python-dotenv, which automatically strips quote
characters from .env values. Docker's --env-file flag does not — it passes
the literal value, quotes included. GROQ_API_KEY="actual_key" worked fine
locally but caused Groq API calls to fail with 401 Invalid API Key inside
the container, since the key string included literal quote marks. Found by
directly inspecting the environment variable inside a throwaway container
(docker run ... python -c "..."), comparing its value against the raw .env
file content. Fixed by removing quotes from .env; python-dotenv handles the
unquoted value identically, so this did not break local runs.
