"""Sanity check: run the trained model on one real test-set receipt and
compare its predictions against ground truth, before scaling to full eval."""
import json
from pathlib import Path

from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.tokenize_boxes import tokenize_box_file
from app.extraction.layoutlm_dataset import normalize_box
from PIL import Image

TEST_BOX_DIR = Path("data/raw/sroie/test/box")
TEST_IMG_DIR = Path("data/raw/sroie/test/img")
TEST_ENTITIES_DIR = Path("data/raw/sroie/test/entities")


def main() -> None:
    box_path = sorted(TEST_BOX_DIR.glob("*.txt"))[0]
    receipt_id = box_path.stem
    image_path = TEST_IMG_DIR / f"{receipt_id}.jpg"
    entity_path = TEST_ENTITIES_DIR / f"{receipt_id}.txt"

    tokens_with_boxes = tokenize_box_file(box_path)
    tokens = [t.text for t in tokens_with_boxes]

    with Image.open(image_path) as img:
        width, height = img.size
    boxes = [normalize_box(t.box, width, height) for t in tokens_with_boxes]

    print("Loading trained model...")
    extractor = LayoutLMExtractor()

    print(f"Running inference on {receipt_id}...")
    predicted = extractor.predict(str(image_path), tokens, boxes)

    ground_truth = {}
    if entity_path.exists():
        ground_truth = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))

    print(f"\n=== {receipt_id} ===")
    for field in ["COMPANY", "DATE", "ADDRESS", "TOTAL"]:
        pred = predicted[field]
        print(f"{field:10} predicted={pred['value']!r}  confidence={pred['confidence']}")
        print(f"{'':10} truth    ={ground_truth.get(field.lower())!r}")


if __name__ == "__main__":
    main()