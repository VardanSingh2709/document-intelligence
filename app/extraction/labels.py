"""The fixed label set for BIO-tagged entity extraction, and the ID mappings
LayoutLMv3 training/inference requires. This file must never change once
training begins, or saved model predictions become meaningless."""

LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL", "I-TOTAL",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}