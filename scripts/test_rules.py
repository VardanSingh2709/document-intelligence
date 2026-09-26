"""Sanity-test rule-based extraction against a few real ground-truth entity files
before running the full evaluation."""
import json
from pathlib import Path

from app.extraction.rules import extract_date, extract_total
from app.ocr.ground_truth import load_ground_truth_lines, load_ground_truth_text

BOX_DIR = Path("data/raw/sroie/train/box")
ENTITIES_DIR = Path("data/raw/sroie/train/entities")
SAMPLE_IDS = ["X00016469612", "X00016469619", "X00016469620"]


def main() -> None:
    for receipt_id in SAMPLE_IDS:
        ground_truth = json.loads((ENTITIES_DIR / f"{receipt_id}.txt").read_text(encoding="utf-8"))
        flat_text = load_ground_truth_text(BOX_DIR / f"{receipt_id}.txt")
        line_text = load_ground_truth_lines(BOX_DIR / f"{receipt_id}.txt")
        # Using ground-truth OCR text here deliberately: we're testing the RULES
        # in isolation first, before compounding errors with our own OCR's mistakes.

        predicted_date = extract_date(flat_text)
        predicted_total = extract_total(line_text)

        print(f"\n=== {receipt_id} ===")
        print(f"Date:  ground truth={ground_truth['date']!r:15} predicted={predicted_date!r}")
        print(f"Total: ground truth={ground_truth['total']!r:10} predicted={predicted_total!r}")


if __name__ == "__main__":
    main()