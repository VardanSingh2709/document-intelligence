"""Unit tests for shared scoring utilities (Phase 7's app/extraction/scoring.py).
These are pure functions with no external dependencies — fast, deterministic."""
from app.extraction.scoring import normalize_date, normalize_money, score_field, text_similarity


def test_normalize_money_strips_currency_symbols():
    assert normalize_money("RM4.00") == 4.00
    assert normalize_money("$9.00") == 9.00
    assert normalize_money("9.00") == 9.00


def test_normalize_money_handles_none():
    assert normalize_money(None) is None


def test_normalize_money_handles_invalid_input():
    assert normalize_money("not a number") is None


def test_normalize_date_ignores_separator_differences():
    # Different separators should normalize to the same value.
    assert normalize_date("25/12/2018") == normalize_date("25-12-2018")


def test_normalize_date_handles_none():
    assert normalize_date(None) is None


def test_text_similarity_identical_strings():
    assert text_similarity("OJC MARKETING", "OJC MARKETING") == 1.0


def test_text_similarity_case_insensitive():
    assert text_similarity("ojc marketing", "OJC MARKETING") == 1.0


def test_text_similarity_both_none_is_perfect_match():
    # Two missing values should count as agreeing, not as a mismatch.
    assert text_similarity(None, None) == 1.0


def test_text_similarity_one_none_is_zero():
    assert text_similarity("something", None) == 0.0


def test_score_field_perfect_predictions():
    predictions = ["9.00", "60.30"]
    truths = ["9.00", "60.30"]
    result = score_field(predictions, truths)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_score_field_missing_prediction_counts_as_false_negative():
    predictions = [None, "60.30"]
    truths = ["9.00", "60.30"]
    result = score_field(predictions, truths)
    assert result["fn"] == 1
    assert result["tp"] == 1