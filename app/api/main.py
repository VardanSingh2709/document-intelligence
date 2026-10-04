"""FastAPI application entry point. Run with:
    uvicorn app.api.main:app --reload
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.schemas import DocumentUploadResponse
from app.review.database import create_document, init_db

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(
    title="Document Intelligence API",
    description="OCR + LayoutLMv3 + LLM fallback pipeline for receipt field extraction.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    """Ensure the database and upload directory exist before accepting requests."""
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accept an uploaded receipt image, save it, and register it for processing."""
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}")

    document_id = create_document(filename=file.filename, file_path="")  # path set after we know the id
    save_path = UPLOAD_DIR / f"{document_id}{suffix}"

    with open(save_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    from app.review.database import get_connection
    with get_connection() as conn:
        conn.execute("UPDATE documents SET file_path = ? WHERE id = ?", (str(save_path), document_id))

    return DocumentUploadResponse(document_id=document_id, filename=file.filename, status="uploaded")