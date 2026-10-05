"""Pydantic request/response schemas for the API. This is the boundary where
untrusted external input meets our system, so we validate explicitly here
(unlike our internal dataclasses in Phase 6, which only organize trusted,
already-validated data)."""
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str


class FieldResult(BaseModel):
    value: str | None
    source: str
    confidence: float | None


class ProcessResponse(BaseModel):
    document_id: str
    status: str
    fields: dict[str, FieldResult] | None = None
    escalated_fields: list[str] | None = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    fields: dict[str, FieldResult] | None = None