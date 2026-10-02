"""Build BIO-labeled training data for SROIE receipts (train or test split), and
report matching statistics so we know how much real, usable data we have."""
import json
import sys
from pathlib import Path

from app.extraction.bio_labeling import LABEL_FIELDS, apply_bio_labels
from app.extraction.tokenize_boxes import tokenize_box_file


def build_for_split(split: str) -> None:
    box_dir = Path(f"data/raw/sroie/{split}/box")
    entities_dir = Path(f"data/raw/sroie/{split}/entities")
    output_path = Path(f"data/processed/bio_{split}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    match_counts = {field: 0 for field in LABEL_FIELDS}
    total_receipts = 0
    unmatched_examples = {field: [] for field in LABEL_FIELDS}

    with open(output_path, "w", encoding="utf-8") as out_file:
        for box_path in sorted(box_dir.glob("*.txt")):
            receipt_id = box_path.stem
            entity_path = entities_dir / f"{receipt_id}.txt"
            if not entity_path.exists():
                continue  # expected for the 14 test receipts with no Task 3 entity labels

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

    print(f"\n[{split.upper()}] Processed {total_receipts} receipts.")
    print(f"[{split.upper()}] Field match rates:")
    for field in LABEL_FIELDS:
        rate = match_counts[field] / total_receipts if total_receipts else 0
        print(f"  {field:10} {match_counts[field]:4}/{total_receipts}  ({rate:.1%})")

    print(f"[{split.upper()}] Sample unmatched entities:")
    for field in LABEL_FIELDS:
        if unmatched_examples[field]:
            print(f"  {field}:")
            for receipt_id, text in unmatched_examples[field]:
                print(f"    {receipt_id}: {text!r}")

    print(f"[{split.upper()}] Saved to {output_path}")


def main() -> None:
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["train", "test"]
    for split in splits:
        build_for_split(split)


if __name__ == "__main__":
    main()