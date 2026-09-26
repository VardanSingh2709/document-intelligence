"""Run an OCR engine over a small subset of SROIE images and report results."""
from pathlib import Path

from app.ocr.paddle_engine import PaddleOCREngine

TRAIN_IMG_DIR = Path("data/raw/sroie/train/img")
SAMPLE_SIZE = 5


def main() -> None:
    image_paths = sorted(TRAIN_IMG_DIR.glob("*.jpg"))[:SAMPLE_SIZE]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {TRAIN_IMG_DIR}")

    print("Loading PaddleOCR engine...")
    engine = PaddleOCREngine()

    for path in image_paths:
        result = engine.extract(str(path))
        print(f"\n=== {path.name} ===")
        print(f"Engine: {result.engine_name}  Time: {result.inference_seconds:.2f}s  Lines: {len(result.lines)}")
        for line in result.lines[:5]:
            print(f"  {line.confidence:.3f}  {line.text!r}")


if __name__ == "__main__":
    main()