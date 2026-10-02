"""Load our BIO-labeled .jsonl data and convert it into the tensor format
LayoutLMv3 requires: normalized boxes, the document image, and aligned labels."""
import json
from pathlib import Path

from PIL import Image

from app.extraction.labels import LABEL_TO_ID


def normalize_box(box: list[int], image_width: int, image_height: int) -> list[int]:
    """Scale a pixel-coordinate box to LayoutLMv3's required 0-1000 range."""
    x1, y1, x2, y2 = box
    return [
        int(1000 * x1 / image_width),
        int(1000 * y1 / image_height),
        int(1000 * x2 / image_width),
        int(1000 * y2 / image_height),
    ]


def load_examples(jsonl_path: Path, image_dir: Path) -> list[dict]:
    """Read a bio_{split}.jsonl file and attach each record's image and
    normalized boxes, producing examples ready for the LayoutLMv3 processor."""
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_path = image_dir / f"{record['receipt_id']}.jpg"
            if not image_path.exists():
                continue  # shouldn't happen given our Phase 2 verification, but fail safe

            with Image.open(image_path) as img:
                width, height = img.size

            normalized_boxes = [
                normalize_box(box, width, height) for box in record["boxes"]
            ]
            label_ids = [LABEL_TO_ID[label] for label in record["labels"]]

            examples.append({
                "receipt_id": record["receipt_id"],
                "image_path": str(image_path),
                "tokens": record["tokens"],
                "boxes": normalized_boxes,
                "label_ids": label_ids,
            })
    return examples