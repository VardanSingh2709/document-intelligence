"""Run confidence-aware inference on the validation set (through our own OCR)
and analyze how well confidence scores separate correct from incorrect
predictions, to derive per-field acceptance thresholds."""
import json
from pathlib import Path

from PIL import Image

from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.ocr_to_tokens import ocr_result_to_tokens
from app.extraction.scoring import normalize_date, normalize_money, text_similarity
from app.ocr.paddle_engine import PaddleOCREngine

TRAIN_IMG_DIR = Path("data/raw/sroie/train/img")
TRAIN_ENTITIES_DIR = Path("data/raw/sroie/train/entities")
SPLIT_FILE = Path("data/raw/sroie/train_val_split.json")

FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]
# Freeform fields use fuzzy similarity; a similarity above this counts as "correct"
# for this analysis (consistent with how we've treated "close enough" elsewhere).
SIMILARITY_CORRECT_THRESHOLD = 0.90


def is_correct(field: str, predicted_value: str | None, truth_value: str | None) -> bool:
    if field == "TOTAL":
        return normalize_money(predicted_value) == normalize_money(truth_value)
    if field == "DATE":
        return normalize_date(predicted_value) == normalize_date(truth_value)
    return text_similarity(predicted_value, truth_value) >= SIMILARITY_CORRECT_THRESHOLD


def main() -> None:
    val_ids = json.loads(SPLIT_FILE.read_text())["val"]
    print(f"Running confidence analysis on {len(val_ids)} validation receipts...\n")

    print("Loading OCR engine and model...")
    ocr_engine = PaddleOCREngine()
    extractor = LayoutLMExtractor()

    # records[field] = list of (confidence, was_correct) tuples
    records: dict[str, list[tuple[float, bool]]] = {f: [] for f in FIELDS}

    for i, receipt_id in enumerate(val_ids):
        image_path = TRAIN_IMG_DIR / f"{receipt_id}.jpg"
        entity_path = TRAIN_ENTITIES_DIR / f"{receipt_id}.txt"
        if not image_path.exists() or not entity_path.exists():
            continue

        ground_truth = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))

        ocr_result = ocr_engine.extract(str(image_path))
        tokens_with_boxes = ocr_result_to_tokens(ocr_result)
        if not tokens_with_boxes:
            continue

        tokens = [t.text for t in tokens_with_boxes]
        with Image.open(image_path) as img:
            width, height = img.size
        boxes = [normalize_box(t.box, width, height) for t in tokens_with_boxes]

        predicted = extractor.predict(str(image_path), tokens, boxes)

        for field in FIELDS:
            pred = predicted[field]
            if pred["confidence"] is None:
                continue  # model predicted nothing for this field at all
            truth = ground_truth.get(field.lower())
            correct = is_correct(field, pred["value"], truth)
            records[field].append((pred["confidence"], correct))

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(val_ids)} processed")

    # Save raw records for the next step (threshold selection) and for inspection.
    output_path = Path("data/processed/confidence_analysis.json")
    output_path.write_text(json.dumps(records, indent=2))
    print(f"\nSaved raw confidence records to {output_path}")

    print("\n=== Confidence summary by correctness ===")
    for field in FIELDS:
        correct_confs = [c for c, ok in records[field] if ok]
        incorrect_confs = [c for c, ok in records[field] if not ok]
        print(f"\n{field}:")
        print(f"  Correct predictions   (n={len(correct_confs):3}): "
              f"mean conf = {sum(correct_confs)/len(correct_confs):.3f}" if correct_confs else "  Correct predictions   (n=0)")
        print(f"  Incorrect predictions (n={len(incorrect_confs):3}): "
              f"mean conf = {sum(incorrect_confs)/len(incorrect_confs):.3f}" if incorrect_confs else "  Incorrect predictions (n=0)")


if __name__ == "__main__":
    main()