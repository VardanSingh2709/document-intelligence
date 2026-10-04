"""Pydantic request/response schemas for the API. This is the boundary where
untrusted external input meets our system, so we validate explicitly here
(unlike our internal dataclasses in Phase 6, which only organize trusted,
already-validated data)."""
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str