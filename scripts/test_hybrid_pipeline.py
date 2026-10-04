"""Sanity check: run the full hybrid pipeline on one real test receipt."""
import json
from pathlib import Path

from PIL import Image

from app.extraction.hybrid_pipeline import HybridExtractor
from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.llm_fallback import GroqFallback
from app.ocr.ground_truth import load_ground_truth_text
from app.extraction.tokenize_boxes import tokenize_box_file

RECEIPT_ID = "X00016469670"
BOX_PATH = Path(f"data/raw/sroie/test/box/{RECEIPT_ID}.txt")
IMAGE_PATH = Path(f"data/raw/sroie/test/img/{RECEIPT_ID}.jpg")
ENTITY_PATH = Path(f"data/raw/sroie/test/entities/{RECEIPT_ID}.txt")


def main() -> None:
    print("Loading models...")
    model_extractor = LayoutLMExtractor()
    llm_fallback = GroqFallback()
    hybrid = HybridExtractor(model_extractor, llm_fallback)

    tokens_with_boxes = tokenize_box_file(BOX_PATH)
    tokens = [t.text for t in tokens_with_boxes]
    with Image.open(IMAGE_PATH) as img:
        width, height = img.size
    boxes = [normalize_box(t.box, width, height) for t in tokens_with_boxes]
    ocr_text = load_ground_truth_text(BOX_PATH)

    ground_truth = json.loads(ENTITY_PATH.read_text(encoding="utf-8", errors="ignore"))

    result = hybrid.extract(str(IMAGE_PATH), tokens, boxes, ocr_text)

    print(f"\nModel time: {result['model_time']:.2f}s  Fallback time: {result['fallback_time']:.2f}s")
    print(f"Escalated fields: {result['escalated_fields']}\n")

    for field, data in result["fields"].items():
        print(f"{field:10} value={data['value']!r:50} source={data['source']:22} truth={ground_truth.get(field.lower())!r}")


if __name__ == "__main__":
    main()