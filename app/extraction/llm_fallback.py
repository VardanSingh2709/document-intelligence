"""LLM-based fallback extraction: given a receipt's raw OCR text and a
specific field name, ask an LLM to extract that one value. Used when
LayoutLMv3's confidence is too low to trust (Phase 8's routing decision),
or for fields (DATE) where confidence itself is known to be unreliable."""
import json

from groq import Groq

from app.config import GROQ_API_KEY

MODEL_NAME = "openai/gpt-oss-20b"

FIELD_DESCRIPTIONS = {
    "COMPANY": "the name of the business/vendor that issued this receipt",
    "DATE": "the date the receipt was issued, in DD/MM/YYYY format",
    "ADDRESS": "the business address printed on the receipt",
    "TOTAL": "the final total amount paid, as a plain number (e.g. 9.00), after any rounding adjustment",
}


class GroqFallback:
    def __init__(self) -> None:
        self.client = Groq(api_key=GROQ_API_KEY)

    def extract_field(self, field: str, ocr_text: str) -> str | None:
        """Ask the LLM to extract one specific field from the receipt's raw
        OCR text. Returns the extracted value, or None if the model
        indicates the field isn't present/determinable."""
        description = FIELD_DESCRIPTIONS[field]

        prompt = f"""You are extracting structured data from OCR text of a retail receipt. The OCR text may contain minor errors (misread characters, merged words).

Extract: {description}

Receipt OCR text:
---
{ocr_text}
---

Respond with ONLY a JSON object in this exact format, nothing else:
{{"value": "the extracted value"}}

If you cannot determine this field with reasonable confidence, respond with:
{{"value": null}}"""

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )

        raw_output = response.choices[0].message.content.strip()
        return self._parse_response(raw_output)

    def _parse_response(self, raw_output: str) -> str | None:
        """Parse the model's JSON response defensively — LLMs sometimes wrap
        JSON in markdown code fences or add stray text despite instructions."""
        text = raw_output.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        try:
            parsed = json.loads(text)
            return parsed.get("value")
        except (json.JSONDecodeError, AttributeError):
            return None