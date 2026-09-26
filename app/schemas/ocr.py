"""Shared data shapes for OCR output, independent of which engine produced it."""
from dataclasses import dataclass


@dataclass
class OCRLine:
    """One detected line of text."""
    text: str
    confidence: float
    box: list[list[float]]  # four [x, y] corner points, clockwise from top-left


@dataclass
class OCRResult:
    """Everything one OCR engine produced for one image."""
    image_path: str
    engine_name: str
    inference_seconds: float
    lines: list[OCRLine]