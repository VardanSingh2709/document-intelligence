"""The OCR engine interface. Every OCR engine must implement this exactly."""
from abc import ABC, abstractmethod

from app.schemas.ocr import OCRResult


class OCREngine(ABC):
    """Abstract base class: defines what an OCR engine must do, not how."""

    @abstractmethod
    def extract(self, image_path: str) -> OCRResult:
        """Run OCR on one image and return a standardized OCRResult."""
        raise NotImplementedError