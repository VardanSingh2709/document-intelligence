"""Fine-tune LayoutLMv3 on our SROIE BIO-labeled receipt data.
Supports a --trial flag for a fast, tiny sanity-check run before committing
to a full training run."""
import argparse
import json
from pathlib import Path

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    EarlyStoppingCallback,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

from app.extraction.labels import ID_TO_LABEL, LABEL_LIST
from app.extraction.layoutlm_dataset import load_examples
from app.extraction.torch_dataset import ReceiptDataset

TRAIN_JSONL = Path("data/processed/bio_train.jsonl")
TRAIN_IMG_DIR = Path("data/raw/sroie/train/img")
SPLIT_FILE = Path("data/raw/sroie/train_val_split.json")
OUTPUT_DIR = Path("models/layoutlmv3_receipts")


def compute_metrics(eval_pred):
    """Called automatically after each evaluation pass. Converts raw model
    predictions back into BIO label strings and computes seqeval's
    precision/recall/F1, correctly handling B-/I- span logic."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        pred_labels = []
        true_lbls = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:  # skip ignored positions, exactly as training does
                pred_labels.append(ID_TO_LABEL[p])
                true_lbls.append(ID_TO_LABEL[l])
        true_predictions.append(pred_labels)
        true_labels.append(true_lbls)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", action="store_true", help="Tiny run (10 examples, 1 epoch) to sanity-check the pipeline")
    args = parser.parse_args()

    print("Loading processor and model...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=len(LABEL_LIST)
    )

    print("Loading examples...")
    all_examples = load_examples(TRAIN_JSONL, TRAIN_IMG_DIR)
    split = json.loads(SPLIT_FILE.read_text())
    train_ids = set(split["train"])
    val_ids = set(split["val"])

    train_examples = [e for e in all_examples if e["receipt_id"] in train_ids]
    val_examples = [e for e in all_examples if e["receipt_id"] in val_ids]

    if args.trial:
        print("TRIAL MODE: using 10 train / 5 val examples, 1 epoch")
        train_examples = train_examples[:10]
        val_examples = val_examples[:5]
        num_epochs = 1
        output_dir = Path("models/trial_run")
    else:
        num_epochs = 15
        output_dir = OUTPUT_DIR

    print(f"Train examples: {len(train_examples)}  Validation examples: {len(val_examples)}")

    train_dataset = ReceiptDataset(train_examples, processor)
    val_dataset = ReceiptDataset(val_examples, processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",  # disable wandb/other auto-logging we haven't set up
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    print(f"\nSaving final model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()