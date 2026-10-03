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

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = probs.argmax(-1).squeeze().tolist()
        confidences = probs.max(-1).values.squeeze().tolist()
        word_ids = encoding_cpu.word_ids(batch_index=0)

        return self._reconstruct_fields(tokens, predictions, confidences, word_ids)

    def _reconstruct_fields(
        self, tokens: list[str], predictions: list[int], confidences: list[float], word_ids: list
    ) -> dict[str, dict]:
        """Walk through sub-token predictions, keep only the first sub-token's
        prediction per word, group consecutive same-field tokens into field
        strings, and compute each field's confidence as the MINIMUM token
        confidence in its span (a single weak/garbled token should pull the
        whole field's confidence down — see Phase 8 write-up)."""
        word_predictions: dict[int, tuple[str, float]] = {}
        seen_words = set()
        for pred_id, conf, word_id in zip(predictions, confidences, word_ids):
            if word_id is None or word_id in seen_words:
                continue
            seen_words.add(word_id)
            word_predictions[word_id] = (ID_TO_LABEL[pred_id], conf)

        fields: dict[str, list[tuple[str, float]]] = {"COMPANY": [], "DATE": [], "ADDRESS": [], "TOTAL": []}
        for word_id in sorted(word_predictions):
            label, conf = word_predictions[word_id]
            if label == "O":
                continue
            field = label.split("-", 1)[1]
            if field in fields:
                fields[field].append((tokens[word_id], conf))

        result = {}
        for field, word_conf_pairs in fields.items():
            if not word_conf_pairs:
                result[field] = {"value": None, "confidence": None}
            else:
                text = _clean_field_text(" ".join(w for w, _ in word_conf_pairs))
                min_confidence = min(c for _, c in word_conf_pairs)
                result[field] = {"value": text, "confidence": min_confidence}
        return result


def _clean_field_text(text: str) -> str:
    """Strip leading punctuation/colons that sometimes get bundled into the
    first token of a predicted span (e.g., a label's trailing ':' merged
    with the value during OCR tokenization)."""
    return text.lstrip(": ").strip()