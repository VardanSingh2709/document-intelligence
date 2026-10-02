"""Sanity check: tokenize one receipt's box file and inspect the word-level output."""
from pathlib import Path

from app.extraction.tokenize_boxes import tokenize_box_file

BOX_PATH = Path("data/raw/sroie/train/box/X00016469612.txt")


def main() -> None:
    tokens = tokenize_box_file(BOX_PATH)
    print(f"Total word tokens: {len(tokens)}\n")
    for token in tokens[:15]:
        print(f"{token.text!r:20} box={token.box}")


if __name__ == "__main__":
    main()