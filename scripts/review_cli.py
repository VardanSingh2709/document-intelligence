"""Simple command-line human review tool. Shows pending review items one at
a time with enough context to make a decision, and lets the reviewer accept,
edit, or reject each one. This is the Phase 10 human-in-the-loop interface;
a web UI (Phase 12) will later call the same resolve_item() function."""
from pathlib import Path

from app.ocr.ground_truth import load_ground_truth_text
from app.review.database import get_pending_items, resolve_item

TRAIN_BOX_DIR = Path("data/raw/sroie/train/box")
TEST_BOX_DIR = Path("data/raw/sroie/test/box")


def find_box_file(receipt_id: str) -> Path | None:
    """Our review items could come from either split; check both."""
    for box_dir in (TRAIN_BOX_DIR, TEST_BOX_DIR):
        candidate = box_dir / f"{receipt_id}.txt"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    pending = get_pending_items()
    if not pending:
        print("No pending review items. Nothing to do.")
        return

    print(f"{len(pending)} item(s) pending review.\n")

    for item in pending:
        print("=" * 60)
        print(f"Receipt:    {item['receipt_id']}")
        print(f"Field:      {item['field']}")
        print(f"Model said: {item['model_value']!r} (confidence: {item['model_confidence']})")
        print(f"LLM said:   {item['fallback_value']!r}")

        box_path = find_box_file(item["receipt_id"])
        if box_path:
            print(f"\nRaw receipt text:\n{load_ground_truth_text(box_path)}\n")
        else:
            print("\n(Could not locate raw OCR text for context.)\n")

        print("Options: [a]ccept shown value, [e]dit, [r]eject, [s]kip for now")
        choice = input("> ").strip().lower()

        if choice == "a":
            value = item["model_value"] or item["fallback_value"]
            resolve_item(item["id"], "accepted", value)
            print(f"Accepted: {value!r}")
        elif choice == "e":
            value = input("Enter the correct value: ").strip()
            resolve_item(item["id"], "edited", value)
            print(f"Saved edited value: {value!r}")
        elif choice == "r":
            resolve_item(item["id"], "rejected", None)
            print("Marked as rejected (no usable value).")
        else:
            print("Skipped.")

    print("\nReview session complete.")


if __name__ == "__main__":
    main()