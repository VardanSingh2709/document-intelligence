"""Sanity check: run the LLM fallback on one real receipt's OCR text for
each field, before wiring it into the full routing pipeline."""
from pathlib import Path

from app.extraction.llm_fallback import GroqFallback
from app.ocr.ground_truth import load_ground_truth_text
import json

BOX_PATH = Path("data/raw/sroie/test/box/X00016469670.txt")
ENTITY_PATH = Path("data/raw/sroie/test/entities/X00016469670.txt")


def main() -> None:
    ocr_text = load_ground_truth_text(BOX_PATH)
    ground_truth = json.loads(ENTITY_PATH.read_text(encoding="utf-8", errors="ignore"))

    print("Loading Groq fallback engine...")
    fallback = GroqFallback()

    for field in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
        predicted = fallback.extract_field(field, ocr_text)
        print(f"\n{field}:")
        print(f"  LLM predicted: {predicted!r}")
        print(f"  Ground truth:  {ground_truth.get(field.lower())!r}")


if __name__ == "__main__":
    main()