"""Score OCR engine output against SROIE ground truth using Character Error Rate."""
from pathlib import Path

import jiwer

from app.ocr.ground_truth import load_ground_truth_text
from app.ocr.tesseract_engine import TesseractEngine

TRAIN_IMG_DIR = Path("data/raw/sroie/train/img")
TRAIN_BOX_DIR = Path("data/raw/sroie/train/box")
SAMPLE_SIZE = 5


def main() -> None:
    image_paths = sorted(TRAIN_IMG_DIR.glob("*.jpg"))[:SAMPLE_SIZE]

    print(f"Loading {TesseractEngine.__name__}...")
    engine = TesseractEngine()

    cers = []
    for image_path in image_paths:
        box_path = TRAIN_BOX_DIR / f"{image_path.stem}.txt"
        ground_truth = load_ground_truth_text(box_path)

        result = engine.extract(str(image_path))
        predicted = " ".join(line.text for line in result.lines)

        # Case-insensitive comparison (see write-up: casing is an engine quirk, not a reading error)
        cer = jiwer.cer(ground_truth.lower(), predicted.lower())
        cers.append(cer)

        print(f"\n=== {image_path.name} ===")
        print(f"CER: {cer:.3f}   (inference: {result.inference_seconds:.2f}s)")
        print(f"Ground truth (first 100 chars): {ground_truth[:100]!r}")
        print(f"Predicted    (first 100 chars): {predicted[:100]!r}")

    avg_cer = sum(cers) / len(cers)
    print(f"\n=== Average CER over {len(cers)} images: {avg_cer:.3f} ===")


if __name__ == "__main__":
    main()