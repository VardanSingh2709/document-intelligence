"""Sanity check: find token spans for one receipt's known entities."""
import json
from pathlib import Path

from app.extraction.bio_labeling import find_best_span
from app.extraction.tokenize_boxes import tokenize_box_file

RECEIPT_ID = "X00016469612"
BOX_PATH = Path(f"data/raw/sroie/train/box/{RECEIPT_ID}.txt")
ENTITY_PATH = Path(f"data/raw/sroie/train/entities/{RECEIPT_ID}.txt")


def main() -> None:
    tokens = tokenize_box_file(BOX_PATH)
    entities = json.loads(ENTITY_PATH.read_text(encoding="utf-8", errors="ignore"))

    for field in ["company", "date", "address", "total"]:
        entity_text = entities.get(field, "")
        span = find_best_span(tokens, entity_text)
        if span:
            start, end = span
            matched_text = " ".join(t.text for t in tokens[start:end])
            print(f"{field.upper():10} ground truth={entity_text!r:50} matched tokens[{start}:{end}] = {matched_text!r}")
        else:
            print(f"{field.upper():10} ground truth={entity_text!r:50} NO MATCH FOUND")


if __name__ == "__main__":
    main()