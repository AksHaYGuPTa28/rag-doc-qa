import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.dependencies import get_pipeline
from app.models.schemas import DocumentSummary, UploadResponse
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...), pipeline: RAGPipeline = Depends(get_pipeline)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}")

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        return pipeline.ingest(str(dest), file.filename)
    except Exception as exc:
        raise HTTPException(500, f"Failed to ingest document: {exc}") from exc


@router.get("", response_model=list[DocumentSummary])
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    return pipeline.list_documents()
