"""Evaluate rule-based date/total extraction against the full validation set.
This is our Phase 4 baseline number: the floor that Phase 6's ML model must beat."""
import json
from pathlib import Path

from app.extraction.rules import extract_date, extract_total
from app.ocr.ground_truth import load_ground_truth_lines, load_ground_truth_text

BOX_DIR = Path("data/raw/sroie/train/box")
ENTITIES_DIR = Path("data/raw/sroie/train/entities")
SPLIT_FILE = Path("data/raw/sroie/train_val_split.json")


def normalize_money(value: str | None) -> float | None:
    """Compare totals as numbers, not strings. Strips currency symbols/codes so
    'RM4.00' and '4.00' are correctly treated as equal amounts."""
    if value is None:
        return None
    import re
    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def score_field(predictions: list, ground_truths: list) -> dict:
    """Compute precision, recall, F1 for one field across all receipts.
    TP: predicted a value and it matches ground truth.
    FP: predicted a value but it's wrong (or ground truth had none).
    FN: predicted nothing but ground truth had a value.
    """
    tp = fp = fn = 0
    for pred, truth in zip(predictions, ground_truths):
        if pred is None and truth is not None:
            fn += 1
        elif pred is not None and truth is None:
            fp += 1
        elif pred is not None and truth is not None:
            if pred == truth:
                tp += 1
            else:
                fp += 1
        # both None: not counted, nothing to measure

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    val_ids = json.loads(SPLIT_FILE.read_text())["val"]
    print(f"Evaluating on {len(val_ids)} validation receipts...\n")

    date_preds, date_truths = [], []
    total_preds, total_truths = [], []
    mismatches = []

    for receipt_id in val_ids:
        entity_path = ENTITIES_DIR / f"{receipt_id}.txt"
        box_path = BOX_DIR / f"{receipt_id}.txt"
        if not entity_path.exists() or not box_path.exists():
            continue

        ground_truth = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))
        flat_text = load_ground_truth_text(box_path)
        line_text = load_ground_truth_lines(box_path)

        pred_date = extract_date(flat_text)
        pred_total_raw = extract_total(line_text)

        truth_date = ground_truth.get("date")
        truth_total_raw = ground_truth.get("total")

        date_preds.append(pred_date)
        date_truths.append(truth_date)

        pred_total = normalize_money(pred_total_raw)
        truth_total = normalize_money(truth_total_raw)
        total_preds.append(pred_total)
        total_truths.append(truth_total)

        if pred_total != truth_total:
            mismatches.append((receipt_id, truth_total_raw, pred_total_raw))

    date_scores = score_field(date_preds, date_truths)
    total_scores = score_field(total_preds, total_truths)

    print("=== DATE ===")
    print(f"Precision: {date_scores['precision']:.3f}  Recall: {date_scores['recall']:.3f}  F1: {date_scores['f1']:.3f}")
    print(f"(TP={date_scores['tp']} FP={date_scores['fp']} FN={date_scores['fn']})")

    print("\n=== TOTAL ===")
    print(f"Precision: {total_scores['precision']:.3f}  Recall: {total_scores['recall']:.3f}  F1: {total_scores['f1']:.3f}")
    print(f"(TP={total_scores['tp']} FP={total_scores['fp']} FN={total_scores['fn']})")

    print(f"\n=== First 15 TOTAL mismatches (for failure analysis) ===")
    for receipt_id, truth, pred in mismatches[:15]:
        print(f"  {receipt_id}: ground truth={truth!r}  predicted={pred!r}")


if __name__ == "__main__":
    main()