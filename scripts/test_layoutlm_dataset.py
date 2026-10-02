"""Sanity check: load one example and verify box normalization and label
mapping look correct before scaling to the full dataset."""
from pathlib import Path

from app.extraction.layoutlm_dataset import load_examples
from app.extraction.labels import ID_TO_LABEL

JSONL_PATH = Path("data/processed/bio_train.jsonl")
IMAGE_DIR = Path("data/raw/sroie/train/img")


def main() -> None:
    examples = load_examples(JSONL_PATH, IMAGE_DIR)
    print(f"Loaded {len(examples)} examples.\n")

    example = examples[0]
    print(f"Receipt: {example['receipt_id']}")
    print(f"Token count: {len(example['tokens'])}\n")

    for i in range(10):
        token = example["tokens"][i]
        box = example["boxes"][i]
        label = ID_TO_LABEL[example["label_ids"][i]]
        print(f"{token!r:20} box={box}  label={label}")


if __name__ == "__main__":
    main()