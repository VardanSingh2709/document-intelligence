"""Verify the integrity of the downloaded SROIE dataset.

Checks, for each split, that every image has a matching OCR box file,
and reports how many have entity (key-information) ground truth.
Exits with an error if any image/box pairing is broken.
"""
from pathlib import Path

DATA_ROOT = Path("data/raw/sroie")

# Known, documented gaps — not bugs. See data/README.md for why these exist.
EXPECTED = {
    "train": {"images": 626, "box": 626, "entities": 626},
    "test": {"images": 360, "box": 360, "entities": 347},
}


def ids_in(folder: Path, suffix: str) -> set[str]:
    return {p.stem for p in folder.glob(f"*{suffix}")}


def verify_split(split: str) -> bool:
    base = DATA_ROOT / split
    img_ids = ids_in(base / "img", ".jpg")
    box_ids = ids_in(base / "box", ".txt")
    ent_ids = ids_in(base / "entities", ".txt")

    print(f"\n=== {split.upper()} ===")
    print(f"Images: {len(img_ids)}  Box: {len(box_ids)}  Entities: {len(ent_ids)}")

    ok = True
    missing_box = img_ids - box_ids
    extra_box = box_ids - img_ids
    missing_entities = img_ids - ent_ids

    if missing_box:
        print(f"ERROR: {len(missing_box)} image(s) missing a box file: {sorted(missing_box)[:5]}")
        ok = False
    if extra_box:
        print(f"ERROR: {len(extra_box)} box file(s) with no image: {sorted(extra_box)[:5]}")
        ok = False

    print(f"Images without entity labels (expected, see data/README.md): {len(missing_entities)}")

    expected = EXPECTED[split]
    if len(img_ids) != expected["images"]:
        print(f"WARNING: expected {expected['images']} images, found {len(img_ids)}")
        ok = False

    return ok


def main() -> None:
    results = [verify_split(split) for split in ("train", "test")]
    if all(results):
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed. See ERROR/WARNING lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()