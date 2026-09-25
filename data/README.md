# Dataset: SROIE 2019

Source: ICDAR 2019 Scanned Receipts OCR and Information Extraction (SROIE) competition.
Official downloads: https://rrc.cvc.uab.es/?ch=13&com=downloads (Google Drive + direct links)
License: CC BY 4.0, as stated on the official downloads page.

Not committed to this repository (see `.gitignore`). To reproduce:
run `python -m scripts.verify_sroie` after placing files under `data/raw/sroie/`
as described below.

## Structure

```text
data/raw/sroie/
├── train/{img,box,entities}/  # 626 receipts each
└── test/{img,box,entities}/   # 360 images, 360 box files, 347 entity files
```

- `img/`: receipt photos (.jpg)
- `box/`: OCR line boxes + transcripts, format: `x1,y1,x2,y2,x3,y3,x4,y4,text`
- `entities/`: key-information labels, JSON: `{"company", "date", "address", "total"}`
  (no coordinates — matching these to OCR boxes is a separate step, see Phase 6)

## Known quirks (verified, not bugs)

- The official Drive folders for `task1train` and `task2train` each contained
  duplicate uploads of some receipts (`(1)`, `(2)`, ... suffixes). Verified
  byte-identical via MD5 hash and deduplicated to the correct 626 unique receipts.
- One orphaned box-annotation file (`X51006619570.txt`) had no matching image
  in the official test folder and was removed.
- Only 347 of the 360 test receipts have entity (Task 3) ground truth. This is
  the competition's own Task 3 subset, not a data error. The remaining 14 are
  usable for OCR-only evaluation.