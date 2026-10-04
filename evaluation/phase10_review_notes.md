# Phase 10: Human-in-the-Loop Review

## Design decisions

**Narrow review trigger.** Only `model_fallback_failed` cases (where both
LayoutLMv3 and the LLM fallback failed to produce a value) are queued for
mandatory review. Flagging every low-confidence or escalated field would
mean reviewing ~50% of receipts (given COMPANY's 48% escalation rate from
Phase 9), defeating the purpose of automation. This threshold is narrow by
design, built from Phase 9's measured escalation rates, not guessed.

**SQLite, not a heavier database.** Review data needs persistence and basic
querying (pending items, status updates) but at a scale (single-digit to
low-hundreds of items) that doesn't justify additional infrastructure.
SQLite is zero-setup, built into Python's standard library, and sufficient.

**CLI before a web UI.** A command-line review tool exercises the full
`resolve_item()` logic without requiring Phase 11's API or Phase 12's
frontend to exist first. The eventual web UI will call the same underlying
function, not duplicate the logic.

## Verified end-to-end

Using a real failure case found by scanning the validation set (not a
fabricated example): receipt X51005711453, field DATE, where both
LayoutLMv3 (model_value=None) and the LLM fallback (fallback_value=None)
failed. The review queue correctly captured this item; the CLI correctly
displayed full context (model/fallback values, raw OCR text); and the
edit path correctly persisted a human-provided correction
(final_value='10/03/2018', status='edited').

## A genuine failure mode discovered through this process

The date on this receipt ("10/03/2018") appears unlabeled, embedded inline
in a transaction-code line ("ORD #07 -REG #19- 10/03/2018 17:24:07"),
rather than in the clearly-labeled format ("DATE: 25/12/2018") our training
data predominantly uses. Neither the model nor the LLM fallback recognized
it. This is a specific, demonstrated gap in training data coverage, not a
generic "the model isn't perfect" statement — useful, concrete material for
Phase 20's failure analysis.

## A known issue flagged during testing, not yet fixed

While testing the review path, a separate bug was observed: a TOTAL
prediction of '3.50 8.50' (two concatenated values) was accepted
automatically at 0.966 confidence on the same receipt. This shows high
model confidence does not guarantee a well-formed span — worth a targeted
fix or at minimum a documented limitation in Phase 20.

## Not yet implemented (deliberately deferred)

- Feeding corrections back into retraining (mentioned in the original
  spec as a future direction). This requires enough accumulated corrections
  to be meaningful, and is better addressed after the system has run against
  more real data, rather than with our current single correction.