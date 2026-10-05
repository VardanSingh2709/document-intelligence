"""FastAPI application entry point. Run with:
    uvicorn app.api.main:app --reload
"""
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.api.schemas import DocumentUploadResponse, DocumentStatusResponse, FieldResult, ProcessResponse
from app.extraction.hybrid_pipeline import HybridExtractor
from app.extraction.layoutlm_dataset import normalize_box
from app.extraction.layoutlm_inference import LayoutLMExtractor
from app.extraction.llm_fallback import GroqFallback
from app.extraction.ocr_to_tokens import ocr_result_to_tokens
from app.ocr.paddle_engine import PaddleOCREngine
from app.review.database import add_review_item, create_document, get_connection, get_document, init_db

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(
    title="Document Intelligence API",
    description="OCR + LayoutLMv3 + LLM fallback pipeline for receipt field extraction.",
    version="0.1.0",
)

# Loaded once at startup, reused across every request — these are expensive
# to initialize (model weights, OCR engine) and must not be recreated per-call.
_pipeline_components: dict = {}


@app.on_event("startup")
def startup() -> None:
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline components (this may take a moment)...")
    ocr_engine = PaddleOCREngine()
    model_extractor = LayoutLMExtractor()
    llm_fallback = GroqFallback()

    def on_needs_review(field, model_value, confidence, fallback_value, document_id=None):
        add_review_item(document_id or "unknown", field, model_value, confidence, fallback_value)

    _pipeline_components["ocr_engine"] = ocr_engine
    _pipeline_components["hybrid"] = HybridExtractor(model_extractor, llm_fallback, on_needs_review=None)
    print("Pipeline ready.")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}")

    document_id = create_document(filename=file.filename, file_path="")
    save_path = UPLOAD_DIR / f"{document_id}{suffix}"

    with open(save_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    with get_connection() as conn:
        conn.execute("UPDATE documents SET file_path = ? WHERE id = ?", (str(save_path), document_id))

    return DocumentUploadResponse(document_id=document_id, filename=file.filename, status="uploaded")


@app.post("/documents/{document_id}/process", response_model=ProcessResponse)
def process_document(document_id: str) -> ProcessResponse:
    """Run the full OCR -> LayoutLMv3 -> routing -> LLM fallback pipeline on
    a previously uploaded document."""
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document found with id {document_id}")

    image_path = doc["file_path"]

    ocr_engine = _pipeline_components["ocr_engine"]
    hybrid = _pipeline_components["hybrid"]

    ocr_result = ocr_engine.extract(image_path)
    tokens_with_boxes = ocr_result_to_tokens(ocr_result)
    if not tokens_with_boxes:
        raise HTTPException(status_code=422, detail="OCR detected no text in this image.")

    tokens = [t.text for t in tokens_with_boxes]
    with Image.open(image_path) as img:
        width, height = img.size
    boxes = [normalize_box(t.box, width, height) for t in tokens_with_boxes]
    ocr_text = " ".join(tokens)

    result = hybrid.extract(image_path, tokens, boxes, ocr_text)

    for field, data in result["fields"].items():
        if data["source"] == "model_fallback_failed":
            add_review_item(document_id, field, data["value"], data["confidence"], None)

    fields_out = {field: FieldResult(**data) for field, data in result["fields"].items()}

    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET results_json = ?, status = 'processed', processed_at = datetime('now') WHERE id = ?",
            (json.dumps(result["fields"]), document_id),
        )

    return ProcessResponse(
        document_id=document_id,
        status="processed",
        fields=fields_out,
        escalated_fields=result["escalated_fields"],
    )


@app.get("/documents/{document_id}/results", response_model=DocumentStatusResponse)
def get_results(document_id: str) -> DocumentStatusResponse:
    """Return whatever has been computed for this document so far."""
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No document found with id {document_id}")

    fields_out = None
    if doc["results_json"]:
        raw_fields = json.loads(doc["results_json"])
        fields_out = {field: FieldResult(**data) for field, data in raw_fields.items()}

    return DocumentStatusResponse(
        document_id=document_id,
        filename=doc["filename"],
        status=doc["status"],
        fields=fields_out,
    )