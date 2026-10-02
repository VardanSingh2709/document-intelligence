"""Verify the LayoutLMv3Processor correctly converts one example into model
input tensors, including correct label alignment for sub-word tokens."""
from pathlib import Path

from PIL import Image
from transformers import LayoutLMv3Processor

from app.extraction.labels import ID_TO_LABEL, LABEL_LIST
from app.extraction.layoutlm_dataset import load_examples

JSONL_PATH = Path("data/processed/bio_train.jsonl")
IMAGE_DIR = Path("data/raw/sroie/train/img")


def main() -> None:
    print("Loading LayoutLMv3 processor (downloads weights on first run)...")
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    examples = load_examples(JSONL_PATH, IMAGE_DIR)
    example = examples[0]

    image = Image.open(example["image_path"]).convert("RGB")

    encoding = processor(
        image,
        example["tokens"],
        boxes=example["boxes"],
        word_labels=example["label_ids"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    print(f"\nInput IDs shape:      {encoding['input_ids'].shape}")
    print(f"Attention mask shape: {encoding['attention_mask'].shape}")
    print(f"Bbox shape:           {encoding['bbox'].shape}")
    print(f"Pixel values shape:   {encoding['pixel_values'].shape}")
    print(f"Labels shape:         {encoding['labels'].shape}")

    print("\n--- First 20 aligned labels (sub-token level) ---")
    labels = encoding["labels"][0][:20].tolist()
    for label_id in labels:
        label_name = ID_TO_LABEL[label_id] if label_id != -100 else "(ignored, -100)"
        print(f"  {label_id:4}  {label_name}")


if __name__ == "__main__":
    main()