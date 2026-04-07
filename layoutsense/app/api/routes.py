"""
API Routes — Supabase-backed
==============================
All endpoints now receive an AsyncSession via FastAPI Depends injection.
The DocumentProcessor methods are async and commit through the session.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DocuParseError,
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.db.models import get_db_session
from app.models.schemas import (
    ErrorResponse,
    ParseResponse,
    RegionLabel,
    StructureResponse,
    UploadResponse,
)
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiter for uploads
limiter = Limiter(key_func=get_remote_address)


def get_processor() -> DocumentProcessor:
    return DocumentProcessor()


def _handle_domain_error(exc: DocuParseError) -> JSONResponse:
    status_code = {
        UnsupportedFileTypeError: 415,
        FileTooLargeError: 413,
        DocumentNotFoundError: 404,
    }.get(type(exc), 400)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document for parsing",
    tags=["Documents"],
)
# TODO: Rate limiting is currently disabled due to incompatibility with slowapi and UploadFile
# A middleware-based approach (e.g., per-IP limits in reverse proxy) is recommended for production
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_processor),
    session: AsyncSession = Depends(get_db_session),
):
    """Handle document upload and validation."""
    content = await file.read()
    mime_type = file.content_type or "application/octet-stream"
    filename = file.filename or "unnamed"

    logger.info("Upload: '%s' (%s, %d B)", filename, mime_type, len(content))
    try:
        result = await processor.upload(session, filename=filename, content=content, mime_type=mime_type)
        return JSONResponse(status_code=201, content=result.model_dump() if hasattr(result, 'model_dump') else result)
    except DocuParseError as exc:
        return _handle_domain_error(exc)


# ── POST /parse ───────────────────────────────────────────────────────────────

@router.post(
    "/parse",
    response_model=ParseResponse,
    summary="Run the OCR + layout pipeline",
    tags=["Documents"],
)
async def parse_document(
    document_id: str = Form(...),
    processor: DocumentProcessor = Depends(get_processor),
    session: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    """
    Executes 3-stage PP-OCR + LightGBM layout classification + optional LayoutLMv3
    re-scoring. Result is persisted to Supabase JSONB for free cache on re-call.
    """
    # Redis cache layer (optional, in front of Supabase)
    redis = getattr(request.app.state, "redis", None) if request else None
    cache_key = f"parse:{document_id}"
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                import json
                return JSONResponse(content=json.loads(cached))
        except Exception as exc:
            logger.warning("Redis read failed: %s", exc)

    try:
        result = await processor.parse(session, document_id)
    except DocuParseError as exc:
        return _handle_domain_error(exc)

    if redis:
        try:
            await redis.setex(cache_key, settings.CACHE_TTL_SECONDS, result.model_dump_json())
        except Exception as exc:
            logger.warning("Redis write failed: %s", exc)

    return result


# ── GET /structure/{document_id} ──────────────────────────────────────────────

@router.get(
    "/structure/{document_id}",
    response_model=StructureResponse,
    summary="Get condensed document structure",
    tags=["Documents"],
)
async def get_structure(
    document_id: str,
    processor: DocumentProcessor = Depends(get_processor),
    session: AsyncSession = Depends(get_db_session),
):
    record = await processor.get_record(session, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    parsed = record.get("parsed")
    if parsed is None:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{document_id}' not yet parsed. Call POST /parse first.",
        )

    # Reconstruct ParseResponse from stored dict
    full = ParseResponse(**parsed)
    headings, paragraphs, tables, figures, reading_order = [], [], [], [], []

    for page in full.pages:
        for region in page.regions:
            reading_order.append(region.region_id)
            if region.label == RegionLabel.HEADING:
                headings.append(region.text)
            elif region.label == RegionLabel.PARAGRAPH:
                paragraphs.append(region.text)
            elif region.label == RegionLabel.TABLE:
                # Part 5 (table reconstruction) enriches this with cell objects
                tables.append([b.text for b in region.text_blocks])
            elif region.label == RegionLabel.FIGURE:
                figures.append(region.bounding_box)

    return StructureResponse(
        document_id=document_id,
        filename=record["filename"],
        headings=headings,
        paragraphs=paragraphs,
        tables=tables,
        figures=figures,
        reading_order=reading_order,
    )


# ── GET /documents/{document_id} ──────────────────────────────────────────────

@router.get("/documents/{document_id}", tags=["Documents"])
async def get_document_info(
    document_id: str,
    processor: DocumentProcessor = Depends(get_processor),
    session: AsyncSession = Depends(get_db_session),
):
    record = await processor.get_record(session, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {
        "document_id": document_id,
        "filename": record["filename"],
        "mime_type": record["mime_type"],
        "size_bytes": record["size_bytes"],
        "total_pages": record["total_pages"],
        "is_parsed": record["is_parsed"],
        "parse_status": record["parse_status"],
    }


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
async def delete_document(
    document_id: str,
    processor: DocumentProcessor = Depends(get_processor),
    session: AsyncSession = Depends(get_db_session),
):
    deleted = await processor.invalidate(session, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")


# ── GET /structure/{document_id}/tables ───────────────────────────────────────

@router.get(
    "/structure/{document_id}/tables",
    summary="Get structured table grids with row/column cell objects",
    tags=["Documents"],
)
async def get_table_structure(
    document_id: str,
    processor: DocumentProcessor = Depends(get_processor),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Returns all TABLE regions reconstructed as row/column grids.

    Each cell carries: row index, col index, text, is_header flag,
    and bounding box. Suitable for rendering as HTML tables or
    importing into spreadsheet tools.

    Example cell:
        {"row": 0, "col": 1, "text": "Unit Price", "is_header": true,
         "bounding_box": {"x_min": 420, "y_min": 210, ...}}
    """
    from app.services.table_reconstructor import reconstruct_all_tables
    from app.models.schemas import (
        ReconstructedTableSchema, TableCellSchema, StructureResponseV2
    )

    record = await processor.get_record(session, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    parsed = record.get("parsed")
    if parsed is None:
        raise HTTPException(status_code=409, detail="Document not yet parsed. Call POST /parse first.")

    full = ParseResponse(**parsed)
    all_tables = []

    for page in full.pages:
        tables = reconstruct_all_tables(page.regions)
        for region_id, table in tables.items():
            cells = [
                TableCellSchema(
                    row=c.row, col=c.col, text=c.text,
                    is_header=c.is_header, bounding_box=c.bounding_box,
                )
                for c in table.cells
            ]
            all_tables.append(ReconstructedTableSchema(
                region_id=table.region_id,
                n_rows=table.n_rows,
                n_cols=table.n_cols,
                header_row=table.header_row,
                cells=cells,
                markdown=table.to_markdown(),
            ))

    return {
        "document_id": document_id,
        "filename": record["filename"],
        "n_tables": len(all_tables),
        "tables": [t.model_dump() for t in all_tables],
    }
