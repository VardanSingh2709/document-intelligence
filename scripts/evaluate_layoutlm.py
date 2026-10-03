"""Full, honest evaluation of our trained LayoutLMv3 model on the test set,
per field, directly comparable to Phase 4's rule-based baseline."""
import json
from pathlib import Path

from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.scoring import normalize_date, normalize_money, score_field, text_similarity
from app.extraction.tokenize_boxes import tokenize_box_file
from PIL import Image

TEST_BOX_DIR = Path("data/raw/sroie/test/box")
TEST_IMG_DIR = Path("data/raw/sroie/test/img")
TEST_ENTITIES_DIR = Path("data/raw/sroie/test/entities")


def main() -> None:
    print("Loading trained model...")
    extractor = LayoutLMExtractor()

    box_paths = sorted(TEST_BOX_DIR.glob("*.txt"))
    print(f"Evaluating on {len(box_paths)} test receipts...\n")

    company_sims, address_sims = [], []
    date_preds, date_truths = [], []
    total_preds, total_truths = [], []
    skipped_no_entities = 0

    for i, box_path in enumerate(box_paths):
        receipt_id = box_path.stem
        entity_path = TEST_ENTITIES_DIR / f"{receipt_id}.txt"
        image_path = TEST_IMG_DIR / f"{receipt_id}.jpg"

        if not entity_path.exists():
            skipped_no_entities += 1
            continue  # the known 14 test receipts without Task 3 labels (Phase 2)

        ground_truth = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))

        tokens_with_boxes = tokenize_box_file(box_path)
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

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(box_paths)} processed")

    date_scores = score_field(date_preds, date_truths)
    total_scores = score_field(total_preds, total_truths)

    print(f"\n(Skipped {skipped_no_entities} receipts with no Task 3 ground truth, as expected)\n")

    print("=== LayoutLMv3 Test Set Results ===")
    print(f"DATE     F1: {date_scores['f1']:.3f}  (P: {date_scores['precision']:.3f}  R: {date_scores['recall']:.3f})")
    print(f"TOTAL    F1: {total_scores['f1']:.3f}  (P: {total_scores['precision']:.3f}  R: {total_scores['recall']:.3f})")
    print(f"COMPANY  fuzzy similarity: {sum(company_sims) / len(company_sims):.3f}")
    print(f"ADDRESS  fuzzy similarity: {sum(address_sims) / len(address_sims):.3f}")


if __name__ == "__main__":
    main()