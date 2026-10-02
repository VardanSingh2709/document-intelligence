"""Build BIO-labeled training data for every SROIE training receipt, and
report matching statistics so we know how much real, usable training data
we actually have before proceeding to Phase 6's model training step."""
import json
from pathlib import Path

from app.extraction.bio_labeling import LABEL_FIELDS, apply_bio_labels
from app.extraction.tokenize_boxes import tokenize_box_file

BOX_DIR = Path("data/raw/sroie/train/box")
ENTITIES_DIR = Path("data/raw/sroie/train/entities")
OUTPUT_PATH = Path("data/processed/bio_train.jsonl")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    match_counts = {field: 0 for field in LABEL_FIELDS}
    total_receipts = 0
    unmatched_examples = {field: [] for field in LABEL_FIELDS}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        for box_path in sorted(BOX_DIR.glob("*.txt")):
            receipt_id = box_path.stem
            entity_path = ENTITIES_DIR / f"{receipt_id}.txt"
            if not entity_path.exists():
                continue

            entities = json.loads(entity_path.read_text(encoding="utf-8", errors="ignore"))
            tokens = tokenize_box_file(box_path)
            if not tokens:
                continue

            labeled_tokens, matched = apply_bio_labels(tokens, entities)
            total_receipts += 1

            for field, was_matched in matched.items():
                if was_matched:
                    match_counts[field] += 1
                elif len(unmatched_examples[field]) < 5:
                    unmatched_examples[field].append((receipt_id, entities.get(field.lower())))

            record = {
                "receipt_id": receipt_id,
                "tokens": [t.text for t in labeled_tokens],
                "boxes": [t.box for t in labeled_tokens],
                "labels": [t.label for t in labeled_tokens],
            }
            out_file.write(json.dumps(record) + "\n")

    print(f"Processed {total_receipts} receipts.\n")
    print("=== Field match rates ===")
    for field in LABEL_FIELDS:
        rate = match_counts[field] / total_receipts
        print(f"{field:10} {match_counts[field]:4}/{total_receipts}  ({rate:.1%})")

    print("\n=== Sample unmatched entities (for failure analysis) ===")
    for field in LABEL_FIELDS:
        if unmatched_examples[field]:
            print(f"\n{field}:")
            for receipt_id, text in unmatched_examples[field]:
                print(f"  {receipt_id}: {text!r}")

    print(f"\nSaved labeled data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()