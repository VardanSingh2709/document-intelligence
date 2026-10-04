"""Sanity check: run the hybrid pipeline on a known failure case (both model
and LLM fallback failed for DATE on this receipt) and confirm it correctly
lands in the review queue."""
from pathlib import Path

from PIL import Image

from app.extraction.hybrid_pipeline import HybridExtractor
from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.llm_fallback import GroqFallback
from app.extraction.ocr_to_tokens import ocr_result_to_tokens
from app.ocr.paddle_engine import PaddleOCREngine
from app.review.database import add_review_item, get_pending_items, init_db

RECEIPT_ID = "X51005711453"
IMAGE_PATH = Path(f"data/raw/sroie/train/img/{RECEIPT_ID}.jpg")


def main() -> None:
    init_db()

    def on_needs_review(field, model_value, confidence, fallback_value):
        print(f"  [REVIEW NEEDED] field={field} model_value={model_value!r} confidence={confidence}")
        add_review_item(RECEIPT_ID, field, model_value, confidence, fallback_value)

    print("Loading OCR, models...")
    ocr_engine = PaddleOCREngine()
    model_extractor = LayoutLMExtractor()
    llm_fallback = GroqFallback()
    hybrid = HybridExtractor(model_extractor, llm_fallback, on_needs_review=on_needs_review)

    ocr_result = ocr_engine.extract(str(IMAGE_PATH))
    tokens_with_boxes = ocr_result_to_tokens(ocr_result)
    tokens = [t.text for t in tokens_with_boxes]
    with Image.open(IMAGE_PATH) as img:
        width, height = img.size
    boxes = [normalize_box(t.box, width, height) for t in tokens_with_boxes]
    ocr_text = " ".join(t.text for t in tokens_with_boxes)

    result = hybrid.extract(str(IMAGE_PATH), tokens, boxes, ocr_text)
    print(f"\nEscalated: {result['escalated_fields']}")
    print(f"Fields: {result['fields']}")

    pending = get_pending_items()
    print(f"\nPending review items in database: {len(pending)}")
    for item in pending:
        print(f"  {item}")


if __name__ == "__main__":
    main()