"""PaddleOCR implementation of the OCREngine interface."""
import time

from paddleocr import PaddleOCR

from app.ocr.base import OCREngine
from app.schemas.ocr import OCRLine, OCRResult


class PaddleOCREngine(OCREngine):
    def __init__(self) -> None:
        # Loaded once, reused across every call to extract().
        self._ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def extract(self, image_path: str) -> OCRResult:
        start = time.perf_counter()
        raw_result = self._ocr.ocr(image_path, cls=True)
        elapsed = time.perf_counter() - start

        lines = [
            OCRLine(text=text, confidence=confidence, box=box)
            for box, (text, confidence) in raw_result[0]
        ]

        return OCRResult(
            image_path=image_path,
            engine_name="paddleocr",
            inference_seconds=elapsed,
            lines=lines,
        )