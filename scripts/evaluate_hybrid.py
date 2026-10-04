"""Full hybrid pipeline evaluation on the validation set: measures accuracy,
escalation rates, and latency, compared against the LayoutLMv3-only baseline
from Phase 7. This is the real test of whether confidence-based routing and
LLM fallback (Phases 8-9) actually help."""
import json
import time
from pathlib import Path

from PIL import Image

from app.extraction.hybrid_pipeline import HybridExtractor
from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.llm_fallback import GroqFallback
from app.extraction.ocr_to_tokens import ocr_result_to_tokens
from app.extraction.scoring import normalize_date, normalize_money, score_field, text_similarity
from app.ocr.paddle_engine import PaddleOCREngine

TRAIN_IMG_DIR = Path("data/raw/sroie/train/img")
TRAIN_ENTITIES_DIR = Path("data/raw/sroie/train/entities")
SPLIT_FILE = Path("data/raw/sroie/train_val_split.json")

FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]


def main() -> None:
    val_ids = json.loads(SPLIT_FILE.read_text())["val"]
    print(f"Evaluating hybrid pipeline on {len(val_ids)} validation receipts...\n")

    print("Loading OCR engine, model, and LLM fallback...")
    ocr_engine = PaddleOCREngine()
    model_extractor = LayoutLMExtractor()
    llm_fallback = GroqFallback()
    hybrid = HybridExtractor(model_extractor, llm_fallback)

    company_sims, address_sims = [], []
    date_preds, date_truths = [], []
    total_preds, total_truths = [], []
    escalation_counts = {f: 0 for f in FIELDS}
    total_model_time, total_fallback_time = 0.0, 0.0
    processed = 0

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
        ocr_text = " ".join(t.text for t in tokens_with_boxes)

        result = hybrid.extract(str(image_path), tokens, boxes, ocr_text)
        processed += 1

        for field in result["escalated_fields"]:
            escalation_counts[field] += 1
        total_model_time += result["model_time"]
        total_fallback_time += result["fallback_time"]

        fields = result["fields"]
        company_sims.append(text_similarity(fields["COMPANY"]["value"], ground_truth.get("company")))
        address_sims.append(text_similarity(fields["ADDRESS"]["value"], ground_truth.get("address")))
        date_preds.append(normalize_date(fields["DATE"]["value"]))
        date_truths.append(normalize_date(ground_truth.get("date")))
        total_preds.append(normalize_money(fields["TOTAL"]["value"]))
        total_truths.append(normalize_money(ground_truth.get("total")))

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(val_ids)} processed")

    date_scores = score_field(date_preds, date_truths)
    total_scores = score_field(total_preds, total_truths)

    print(f"\n=== Hybrid Pipeline Results (n={processed}) ===")
    print(f"DATE     F1: {date_scores['f1']:.3f}")
    print(f"TOTAL    F1: {total_scores['f1']:.3f}")
    print(f"COMPANY  fuzzy similarity: {sum(company_sims)/len(company_sims):.3f}")
    print(f"ADDRESS  fuzzy similarity: {sum(address_sims)/len(address_sims):.3f}")

    print(f"\n=== Escalation rates ===")
    for field in FIELDS:
        print(f"  {field:10} {escalation_counts[field]:3}/{processed} ({escalation_counts[field]/processed:.1%})")

    print(f"\n=== Timing (total across {processed} receipts) ===")
    print(f"  Model time:    {total_model_time:.1f}s  (avg {total_model_time/processed:.2f}s/receipt)")
    print(f"  Fallback time: {total_fallback_time:.1f}s  (avg {total_fallback_time/processed:.2f}s/receipt, only when escalated)")


if __name__ == "__main__":
    main()