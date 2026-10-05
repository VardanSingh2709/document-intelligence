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
