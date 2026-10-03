"""The full, honest end-to-end evaluation: our own OCR -> LayoutLMv3 -> field
extraction, compared against ground truth. This is the number that reflects
real-world usage, not the ground-truth-OCR-text numbers from Phase 7's
earlier evaluation."""
import argparse
import json
from pathlib import Path

from PIL import Image

from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.ocr_to_tokens import ocr_result_to_tokens
from app.extraction.scoring import normalize_date, normalize_money, score_field, text_similarity
from app.ocr.paddle_engine import PaddleOCREngine

TEST_IMG_DIR = Path("data/raw/sroie/test/img")
TEST_ENTITIES_DIR = Path("data/raw/sroie/test/entities")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0, help="Run on only N receipts first, as a sanity check")
    args = parser.parse_args()

    print("Loading OCR engine and trained model...")
    ocr_engine = PaddleOCREngine()
    extractor = LayoutLMExtractor()

    image_paths = sorted(TEST_IMG_DIR.glob("*.jpg"))
    if args.trial:
        image_paths = image_paths[: args.trial]
    print(f"Evaluating on {len(image_paths)} test receipts (OCR + LayoutLMv3)...\n")

    company_sims, address_sims = [], []
    date_preds, date_truths = [], []
    total_preds, total_truths = [], []
    skipped_no_entities = 0
    skipped_no_tokens = 0

    for i, image_path in enumerate(image_paths):
        receipt_id = image_path.stem
        entity_path = TEST_ENTITIES_DIR / f"{receipt_id}.txt"
        if not entity_path.exists():
            skipped_no_entities += 1
            continue

        ground_truth = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))

        ocr_result = ocr_engine.extract(str(image_path))
        tokens_with_boxes = ocr_result_to_tokens(ocr_result)
        if not tokens_with_boxes:
            skipped_no_tokens += 1
            continue

        tokens = [t.text for t in tokens_with_boxes]
        with Image.open(image_path) as img:
            width, height = img.size
        boxes = [normalize_box(t.box, width, height) for t in tokens_with_boxes]

        predicted = extractor.predict(str(image_path), tokens, boxes)

        company_sims.append(text_similarity(predicted["COMPANY"], ground_truth.get("company")))
        address_sims.append(text_similarity(predicted["ADDRESS"], ground_truth.get("address")))
        date_preds.append(normalize_date(predicted["DATE"]))
        date_truths.append(normalize_date(ground_truth.get("date")))
        total_preds.append(normalize_money(predicted["TOTAL"]))
        total_truths.append(normalize_money(ground_truth.get("total")))

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(image_paths)} processed")

    date_scores = score_field(date_preds, date_truths)
    total_scores = score_field(total_preds, total_truths)

    print(f"\n(Skipped {skipped_no_entities} with no entity ground truth, {skipped_no_tokens} with no OCR tokens detected)\n")
    print("=== End-to-End (Our OCR + LayoutLMv3) Test Set Results ===")
    print(f"DATE     F1: {date_scores['f1']:.3f}  (P: {date_scores['precision']:.3f}  R: {date_scores['recall']:.3f})")
    print(f"TOTAL    F1: {total_scores['f1']:.3f}  (P: {total_scores['precision']:.3f}  R: {total_scores['recall']:.3f})")
    print(f"COMPANY  fuzzy similarity: {sum(company_sims) / len(company_sims):.3f}")
    print(f"ADDRESS  fuzzy similarity: {sum(address_sims) / len(address_sims):.3f}")


if __name__ == "__main__":
    main()