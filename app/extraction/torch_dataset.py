"""A PyTorch Dataset wrapping our processed SROIE examples, applying the
LayoutLMv3Processor on-the-fly for each example (keeps memory usage low —
we don't pre-process and store all 553 examples' image tensors at once)."""
from PIL import Image
from torch.utils.data import Dataset


class ReceiptDataset(Dataset):
    def __init__(self, examples: list[dict], processor):
        self.examples = examples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        image = Image.open(example["image_path"]).convert("RGB")

        encoding = self.processor(
            image,
            example["tokens"],
            boxes=example["boxes"],
            word_labels=example["label_ids"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        # Processor adds a batch dimension (shape [1, ...]); squeeze it out,
        # since the training loop will add its own batch dimension when
        # grouping multiple examples together.
        return {k: v.squeeze(0) for k, v in encoding.items()}