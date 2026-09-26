"""Tesseract implementation of the OCREngine interface."""
import time

import pytesseract
from PIL import Image

from app.ocr.base import OCREngine
from app.schemas.ocr import OCRLine, OCRResult

# Explicit path, not relying on system PATH (see write-up: explicit config over hidden assumptions).
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractEngine(OCREngine):
    def extract(self, image_path: str) -> OCRResult:
        image = Image.open(image_path)

        start = time.perf_counter()
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        elapsed = time.perf_counter() - start

        lines = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            confidence = float(data["conf"][i]) / 100.0  # Tesseract gives 0-100, we standardize to 0-1
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            lines.append(OCRLine(text=text, confidence=max(confidence, 0.0), box=box))

        return OCRResult(
            image_path=image_path,
            engine_name="tesseract",
            inference_seconds=elapsed,
            lines=lines,
        )