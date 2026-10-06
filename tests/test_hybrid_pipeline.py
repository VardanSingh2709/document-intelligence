"""Tests for the hybrid extraction pipeline's routing logic, using mocked
model/fallback components so we test the WIRING, not the real ML model or
a real network call to Groq."""
from app.extraction.hybrid_pipeline import HybridExtractor


def make_mock_model(mocker, predictions: dict):
    """A fake model_extractor whose .predict() always returns a fixed,
    known set of field predictions, regardless of input."""
    mock = mocker.Mock()
    mock.predict.return_value = predictions
    return mock


def make_mock_fallback(mocker, return_value):
    """A fake llm_fallback whose .extract_field() always returns a fixed value."""
    mock = mocker.Mock()
    mock.extract_field.return_value = return_value
    return mock


def test_high_confidence_field_is_accepted_without_escalation(mocker):
    model = make_mock_model(mocker, {
        "TOTAL": {"value": "9.00", "confidence": 0.99},
    })
    fallback = make_mock_fallback(mocker, "should not be called")

    pipeline = HybridExtractor(model, fallback)
    result = pipeline.extract("fake_path.jpg", ["token"], [[0, 0, 1, 1]], "fake ocr text")

    assert result["fields"]["TOTAL"]["source"] == "model"
    assert result["fields"]["TOTAL"]["value"] == "9.00"
    fallback.extract_field.assert_not_called()  # the whole point of routing: don't call the LLM unnecessarily


def test_low_confidence_field_escalates_and_uses_fallback_value(mocker):
    model = make_mock_model(mocker, {
        "TOTAL": {"value": "9.00", "confidence": 0.50},  # below threshold
    })
    fallback = make_mock_fallback(mocker, "9.50")  # LLM "corrects" it

    pipeline = HybridExtractor(model, fallback)
    result = pipeline.extract("fake_path.jpg", ["token"], [[0, 0, 1, 1]], "fake ocr text")

    assert result["fields"]["TOTAL"]["source"] == "llm_fallback"
    assert result["fields"]["TOTAL"]["value"] == "9.50"
    assert "TOTAL" in result["escalated_fields"]


def test_date_always_escalates_regardless_of_confidence(mocker):
    # Phase 8 finding: DATE's confidence doesn't correlate with correctness,
    # so it must ALWAYS escalate, even at very high confidence.
    model = make_mock_model(mocker, {
        "DATE": {"value": "25/12/2018", "confidence": 0.99},
    })
    fallback = make_mock_fallback(mocker, "25/12/2018")

    pipeline = HybridExtractor(model, fallback)
    result = pipeline.extract("fake_path.jpg", ["token"], [[0, 0, 1, 1]], "fake ocr text")

    assert "DATE" in result["escalated_fields"]
    fallback.extract_field.assert_called_once()


def test_when_both_model_and_fallback_fail_source_is_labeled_correctly(mocker):
    """Regression test for a real bug (Phase 12): the else-branch once
    incorrectly labeled this case as source='llm_fallback' instead of
    'model_fallback_failed', and discarded the model's last-attempt value."""
    model = make_mock_model(mocker, {
        "DATE": {"value": None, "confidence": None},
    })
    fallback = make_mock_fallback(mocker, None)  # LLM also fails

    pipeline = HybridExtractor(model, fallback)
    result = pipeline.extract("fake_path.jpg", ["token"], [[0, 0, 1, 1]], "fake ocr text")

    assert result["fields"]["DATE"]["source"] == "model_fallback_failed"


def test_review_callback_fires_only_when_both_fail(mocker):
    model = make_mock_model(mocker, {
        "DATE": {"value": None, "confidence": None},
    })
    fallback = make_mock_fallback(mocker, None)
    review_callback = mocker.Mock()

    pipeline = HybridExtractor(model, fallback, on_needs_review=review_callback)
    pipeline.extract("fake_path.jpg", ["token"], [[0, 0, 1, 1]], "fake ocr text")

    review_callback.assert_called_once()