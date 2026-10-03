"""Bridge between our OCR engine's output (OCRResult) and the word-level
token format our LayoutLMv3 pipeline expects, reusing the same proportional
word-splitting logic from tokenize_boxes.py (originally built for SROIE's
ground-truth box files, now applied to our own OCR's real output)."""
from app.extraction.tokenize_boxes import WordToken, split_line_into_words
from app.schemas.ocr import OCRResult


def ocr_result_to_tokens(result: OCRResult) -> list[WordToken]:
    """Convert OCR engine output into word-level tokens with axis-aligned boxes."""
    all_tokens = []
    for line in result.lines:
        if not line.text.strip():
            continue
        xs = [point[0] for point in line.box]
        ys = [point[1] for point in line.box]
        axis_box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        all_tokens.extend(split_line_into_words(axis_box, line.text))
    return all_tokens