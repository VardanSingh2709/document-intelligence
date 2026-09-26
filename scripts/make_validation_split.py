"""Carve a validation set out of SROIE's training data, grouped by company
to avoid the same store's receipts appearing in both splits (data leakage).
"""
import json
import random
from pathlib import Path

TRAIN_ENTITIES_DIR = Path("data/raw/sroie/train/entities")
OUTPUT_PATH = Path("data/raw/sroie/train_val_split.json")
VAL_FRACTION = 0.10
SEED = 42  # fixed seed: makes the split reproducible every time this runs


def main() -> None:
    receipt_to_company: dict[str, str] = {}
    for entity_file in TRAIN_ENTITIES_DIR.glob("*.txt"):
        try:
            data = json.loads(entity_file.read_text(encoding="utf-8", errors="ignore"))
            company = data.get("company", "UNKNOWN").strip().upper()
        except json.JSONDecodeError:
            company = "UNKNOWN"
        receipt_to_company[entity_file.stem] = company

    companies = sorted(set(receipt_to_company.values()))
    rng = random.Random(SEED)
    rng.shuffle(companies)

    val_company_count = max(1, int(len(companies) * VAL_FRACTION))
    val_companies = set(companies[:val_company_count])

    train_ids = [rid for rid, c in receipt_to_company.items() if c not in val_companies]
    val_ids = [rid for rid, c in receipt_to_company.items() if c in val_companies]

    print(f"Total receipts: {len(receipt_to_company)}")
    print(f"Unique companies: {len(companies)}")
    print(f"Train: {len(train_ids)} receipts   Validation: {len(val_ids)} receipts")

    OUTPUT_PATH.write_text(json.dumps({"train": train_ids, "val": val_ids}, indent=2))
    print(f"Saved split to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()