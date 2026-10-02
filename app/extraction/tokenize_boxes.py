"""Convert SROIE's line-level OCR boxes into word-level tokens with approximate
per-word bounding boxes, by splitting each line's box proportionally by
character width. This is an approximation (SROIE provides no true word-level
boxes) but is standard practice when only line boxes are available."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WordToken:
    text: str
    box: list[int]  # [x1, y1, x2, y2] — axis-aligned, simplified from the 4-point quad


def parse_box_line(raw_line: str) -> tuple[list[int], str] | None:
    """Parse one line of a SROIE box file: 8 coordinates + transcript.
    Returns an axis-aligned [x1, y1, x2, y2] box (min/max of the 4 points)
    and the transcript, or None if the line is malformed."""
    parts = raw_line.strip().split(",", 8)
    if len(parts) != 9:
        return None
    try:
        coords = [int(p) for p in parts[:8]]
    except ValueError:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    box = [min(xs), min(ys), max(xs), max(ys)]
    return box, parts[8]


def split_line_into_words(box: list[int], text: str) -> list[WordToken]:
    """Split one line's box into per-word boxes, proportional to each word's
    character length relative to the full line's character length."""
    words = text.split()
    if not words:
        return []

    x1, y1, x2, y2 = box
    line_width = x2 - x1
    total_chars = len(text)  # includes spaces, so proportions reflect real spacing

    tokens = []
    char_offset = 0
    for word in words:
        # Find where this word starts within the original text (handles multiple spaces safely).
        start_idx = text.index(word, char_offset)
        end_idx = start_idx + len(word)

        word_x1 = x1 + int(line_width * (start_idx / total_chars))
        word_x2 = x1 + int(line_width * (end_idx / total_chars))
        tokens.append(WordToken(text=word, box=[word_x1, y1, word_x2, y2]))

        char_offset = end_idx
    return tokens


def tokenize_box_file(box_file_path: Path) -> list[WordToken]:
    """Read a full SROIE box file and return word-level tokens for every line."""
    all_tokens = []
    with open(box_file_path, encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            parsed = parse_box_line(raw_line)
            if parsed is None:
                continue
            box, text = parsed
            all_tokens.extend(split_line_into_words(box, text))
    return all_tokens