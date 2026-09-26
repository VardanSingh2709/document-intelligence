"""Load SROIE ground-truth OCR transcripts for comparison against engine output."""
from pathlib import Path


def load_ground_truth_text(box_file_path: Path) -> str:
    """Read a SROIE box/transcript file and return all transcripts joined as one block."""
    lines = []
    with open(box_file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: x1,y1,x2,y2,x3,y3,x4,y4,transcript
            # The transcript itself may contain commas, so split only on the first 8.
            parts = line.split(",", 8)
            if len(parts) == 9:
                lines.append(parts[8])
    return " ".join(lines)


def load_ground_truth_lines(box_file_path) -> list[str]:
    """Read a SROIE box/transcript file and return transcripts as a list of separate lines,
    preserving line boundaries (unlike load_ground_truth_text, which flattens to one block
    for OCR character-error-rate scoring)."""
    lines = []
    with open(box_file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)
            if len(parts) == 9:
                lines.append(parts[8])
    return lines