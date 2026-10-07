# Document Intelligence Platform

An end-to-end system that extracts structured data (vendor, date, address,
total) from photographed receipts, combining OCR, a fine-tuned layout-aware
transformer, confidence-based routing, an LLM fallback, and human review.

Built to measure every design decision against real data rather than
assume it — every number below is reproducible and cited to the evaluation
file that produced it.

![CI](https://github.com/VardanSingh2709/document-intelligence/actions/workflows/ci.yml/badge.svg)

## What it does

Receipt image
|
v
OCR (PaddleOCR)
|
v
LayoutLMv3 (fine-tuned, token classification)
|
v
Confidence-based routing --> high confidence --> accept
|
v
low confidence / DATE (always)
LLM fallback (Groq, Llama-family model)
|
v
still unresolved
Human review queue (SQLite + CLI/web UI)

Try it: upload a receipt through the Streamlit UI, watch it get OCR'd,
classified, and (if needed) escalated to an LLM or flagged for you to
correct — with the source of every value labeled (model / LLM / human).

## Results

The honest, end-to-end numbers (our own OCR feeding the model, not clean
ground-truth text — see [`evaluation/phase19_final_report.md`](evaluation/phase19_final_report.md)
for full methodology and caveats):

| Field   | Rule-based baseline | LayoutLMv3 alone | LayoutLMv3 + LLM fallback |
|---------|---------------------|-------------------|-----------------------------|
| Date    | 0.986 (F1)          | 0.549 (F1)        | **0.950 (F1)**              |
| Total   | 0.554 (F1)          | 0.905 (F1)        | **0.950 (F1)**              |
| Company | 0.848 (similarity)  | 0.885 (similarity)| 0.832 (similarity)          |
| Address | 0.718 (similarity)  | 0.927 (similarity)| 0.902 (similarity)          |

**The headline finding:** real OCR errors collapsed the model's DATE
accuracy from 0.981 (clean text) to 0.549. Analysis of *why* (confidence
scores barely differ between correct and incorrect DATE predictions — see
[`evaluation/phase8_routing_thresholds.md`](evaluation/phase8_routing_thresholds.md))
led to always escalating DATE to an LLM fallback, which recovered accuracy
to 0.950 — a measured fix for a measured, specific weakness, not a blanket
"add an LLM" decision.

**Not every result is a win, and that's documented too:** the LLM fallback
*reduces* accuracy slightly on Company and Address, since it sees only
flattened OCR text and loses the positional signal LayoutLMv3 uses. Full
analysis, including a dangerous high-confidence TOTAL bug found during
testing, is in [`evaluation/phase20_failure_analysis.md`](evaluation/phase20_failure_analysis.md).

## Architecture

- **OCR:** PaddleOCR, benchmarked against Tesseract (13.1% vs 19.1% CER on
  a sample) — [`evaluation/`](evaluation) has the full comparison
- **Extraction model:** LayoutLMv3-base, fine-tuned on SROIE 2019 (626
  receipts), token classification with BIO tagging over 4 entity types
- **Confidence routing:** per-field thresholds derived empirically from a
  held-out validation set, not guessed (0.90 for Total/Address, 0.95 for
  Company, and DATE *always* escalates — its confidence doesn't correlate
  with correctness, see the routing doc above)
- **LLM fallback:** Groq-hosted `openai/gpt-oss-20b`, swappable via a
  small interface (same pattern used for OCR engines)
- **Human review:** SQLite-backed queue, resolvable via CLI or the web UI
- **API:** FastAPI (`app/api/`) — upload, process, retrieve results, submit
  corrections
- **Frontend:** Streamlit — upload, see extracted fields with source/
  confidence, correct anything the system couldn't resolve
- **Tests:** pytest, including regression tests for real bugs found during
  development (`tests/`)
- **Containerized:** Docker (model baked into the image — see
  [known limitations](evaluation/known_issues.md#6-docker-build-is-verified-manually-not-in-ci))
- **CI:** GitHub Actions runs the test suite on every push

## Dataset

[SROIE 2019](https://rrc.cvc.uab.es/?ch=13) (ICDAR Scanned Receipts OCR and
Information Extraction), CC BY 4.0. See [`data/README.md`](data/README.md)
for provenance, known dataset quirks (duplicate uploads, label typos), and
how to reproduce the exact train/val/test split used here.

## Known limitations (see [`evaluation/known_issues.md`](evaluation/known_issues.md) for full detail)

- A span-reconstruction bug can concatenate two values into one TOTAL
  prediction at high confidence — the most operationally concerning issue
  found, since it fails silently rather than triggering review
  ([details](evaluation/phase20_failure_analysis.md#3-wrong-field-association--model-confusion))
- OCR can merge adjacent tokens (e.g., a date and timestamp) into
  unrecoverable garbage; no OCR engine evaluated here fully avoids this
- The trained model (~500MB) is gitignored and baked into the Docker image
  manually; CI runs the test suite only, not a full image build, since the
  model isn't available in a clean CI checkout — a real production
  pipeline would pull it from model storage (e.g., Hugging Face Hub) at
  build time instead

## Running it

**Requirements:** Python 3.12, a Groq API key (free tier — [console.groq.com](https://console.groq.com)),
and optionally a CUDA GPU (falls back to CPU automatically).

```bash
git clone https://github.com/VardanSingh2709/document-intelligence.git
cd document-intelligence
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
cp .env.example .env                 # then add your GROQ_API_KEY
```

You'll also need the trained model in `models/layoutlmv3_receipts/` (not
included in this repo due to size — see [training instructions](#training-from-scratch)
below to produce it yourself).

**Run the API:**
```bash
uvicorn app.api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

**Run the frontend** (separate terminal):
```bash
streamlit run frontend/app.py
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

**Run with Docker:**
```bash
docker build -t document-intelligence-api .
docker run -p 8000:8000 --env-file .env document-intelligence-api
```

## Training from scratch

1. Download SROIE 2019 and verify it: see [`data/README.md`](data/README.md)
2. Build the BIO-labeled dataset: `python -m scripts.build_bio_dataset`
3. Train: `python -m scripts.train_layoutlm` (expects a CUDA GPU; ~25 min
   on an RTX 4050, 6GB VRAM)
4. Evaluate: `python -m scripts.evaluate_layoutlm` /
   `python -m scripts.evaluate_end_to_end` /
   `python -m scripts.evaluate_hybrid`

## Project structure

app/
api/ FastAPI endpoints
extraction/ OCR→tokens, BIO labeling, model inference, routing, LLM fallback
ocr/ Swappable OCR engine interface (PaddleOCR, Tesseract)
review/ SQLite-backed human review queue
schemas/ Shared data shapes
frontend/ Streamlit UI
scripts/ Data prep, training, evaluation (how every result was produced)
tests/ pytest suite
evaluation/ Every measured result, phase by phase, with methodology
data/ Dataset docs (data itself is gitignored — see data/README.md)

## What I'd do next with more time

See [`evaluation/phase20_failure_analysis.md`](evaluation/phase20_failure_analysis.md#summary-what-wed-fix-first-given-limited-time)
for the prioritized list — in short: fix the TOTAL span bug, extend
per-field confidence calibration to Company/Address, and evaluate OCR
engines beyond PaddleOCR/Tesseract, since OCR errors are the root cause
behind most other failures found.