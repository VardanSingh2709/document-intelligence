"""FastAPI application entry point. Run with:
    uvicorn app.api.main:app --reload
"""
from fastapi import FastAPI

app = FastAPI(
    title="Document Intelligence API",
    description="OCR + LayoutLMv3 + LLM fallback pipeline for receipt field extraction.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    """Basic liveness check: confirms the API process is running and responsive."""
    return {"status": "ok"}