"""The full hybrid extraction pipeline: LayoutLMv3 first, then confidence-based
routing (Phase 8) decides which fields get a second opinion from the LLM
fallback (Phase 9). This is the complete system Phase 0 originally envisioned."""
import time

from app.extraction.llm_fallback import GroqFallback
from app.extraction.routing import ALWAYS_ESCALATE_FIELDS, should_accept


class HybridExtractor:
    def __init__(self, model_extractor, llm_fallback: GroqFallback) -> None:
        self.model_extractor = model_extractor
        self.llm_fallback = llm_fallback

    def extract(self, image_path: str, tokens: list[str], boxes: list[list[int]], ocr_text: str) -> dict:
        """Run LayoutLMv3, then escalate any low-confidence or always-escalate
        fields to the LLM fallback. Returns final values plus metadata about
        which fields were escalated and how long each stage took."""
        start = time.perf_counter()
        model_predictions = self.model_extractor.predict(image_path, tokens, boxes)
        model_time = time.perf_counter() - start

        final_results = {}
        fallback_time = 0.0
        escalated_fields = []

        for field, pred in model_predictions.items():
            accepted = should_accept(field, pred["confidence"])

            if accepted:
                final_results[field] = {
                    "value": pred["value"],
                    "source": "model",
                    "confidence": pred["confidence"],
                }
            else:
                escalated_fields.append(field)
                fb_start = time.perf_counter()
                llm_value = self.llm_fallback.extract_field(field, ocr_text)
                fallback_time += time.perf_counter() - fb_start

                final_results[field] = {
                    "value": llm_value if llm_value is not None else pred["value"],
                    "source": "llm_fallback" if llm_value is not None else "model_fallback_failed",
                    "confidence": pred["confidence"],
                }

        return {
            "fields": final_results,
            "escalated_fields": escalated_fields,
            "model_time": model_time,
            "fallback_time": fallback_time,
        }