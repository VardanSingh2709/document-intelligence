"""Load our fine-tuned LayoutLMv3 model and run inference on new token/box/image
inputs, converting raw predictions back into field values for comparison
against ground truth."""
from pathlib import Path

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from app.extraction.labels import ID_TO_LABEL

MODEL_DIR = Path("models/layoutlmv3_receipts")


class LayoutLMExtractor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = LayoutLMv3Processor.from_pretrained(str(MODEL_DIR), apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_DIR)).to(self.device)
        self.model.eval()  # inference mode: disables dropout and other training-only behavior

    def predict(self, image_path: str, tokens: list[str], boxes: list[list[int]]) -> dict[str, str]:
        """Run the model on one receipt's tokens/boxes/image and return
        extracted field values, reconstructed from predicted BIO labels."""
        image = Image.open(image_path).convert("RGB")

        encoding_cpu = self.processor(
            image, tokens, boxes=boxes,
            truncation=True, padding="max_length", max_length=512,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding_cpu.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_ids = encoding_cpu.word_ids(batch_index=0)

        return self._reconstruct_fields(tokens, predictions, word_ids)

    def _reconstruct_fields(self, tokens: list[str], predictions: list[int], word_ids: list) -> dict[str, str]:
        """Walk through sub-token predictions, keep only the first sub-token's
        prediction per word (matching how we trained), and group consecutive
        same-field tokens back into field strings."""
        word_predictions: dict[int, str] = {}
        seen_words = set()
        for pred_id, word_id in zip(predictions, word_ids):
            if word_id is None or word_id in seen_words:
                continue  # skip special tokens and subsequent sub-tokens of an already-seen word
            seen_words.add(word_id)
            word_predictions[word_id] = ID_TO_LABEL[pred_id]

        fields: dict[str, list[str]] = {"COMPANY": [], "DATE": [], "ADDRESS": [], "TOTAL": []}
        for word_id in sorted(word_predictions):
            label = word_predictions[word_id]
            if label == "O":
                continue
            field = label.split("-", 1)[1]  # "B-TOTAL" -> "TOTAL"
            if field in fields:
                fields[field].append(tokens[word_id])

        return {
            field: _clean_field_text(" ".join(words)) if words else None
            for field, words in fields.items()
        }


def _clean_field_text(text: str) -> str:
    """Strip leading punctuation/colons that sometimes get bundled into the
    first token of a predicted span (e.g., a label's trailing ':' merged
    with the value during OCR tokenization)."""
    return text.lstrip(": ").strip()