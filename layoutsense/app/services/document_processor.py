"""
Document Processor — Supabase-backed
======================================
Handles file ingestion, format detection, PDF-to-image conversion,
and orchestrates the full pipeline (OCR → Layout → LayoutLMv3).

Persistence:
  - Document metadata → Supabase `documents` table
  - Parse results → Supabase `parse_jobs` table (JSONB column)
  - Raw page images → process memory only (not persisted; too large)

The in-memory page image cache (_page_cache) is keyed by doc_id and
survives for the lifetime of the process. On restart, metadata is
re-fetched from Supabase but the document must be re-uploaded to
re-parse (images aren't stored). This is the correct trade-off for
a document API: metadata is cheap to store, raw images are not.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    PDFConversionError,
    UnsupportedFileTypeError,
)
from app.db import repository as repo
from app.models.schemas import PageResult, ParseResponse, UploadResponse
from app.services.layout_analyser import LayoutAnalyser
from app.services.ocr_pipeline import OCRPipeline
from app.classifier.layoutlmv3 import get_layoutlmv3_scorer

logger = logging.getLogger(__name__)

# Process-local image cache: doc_id -> (timestamp, List[np.ndarray])
# Cache entries expire after 1 hour to prevent unbounded memory growth
_page_cache: Dict[str, Tuple[float, List[np.ndarray]]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _cleanup_expired_cache():
    """Remove expired cache entries to prevent memory leaks."""
    current_time = time.time()
    expired_keys = [
        doc_id for doc_id, (timestamp, _) in _page_cache.items()
        if current_time - timestamp > _CACHE_TTL_SECONDS
    ]
    for doc_id in expired_keys:
        del _page_cache[doc_id]
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


def _get_cached_pages(doc_id: str) -> Optional[List[np.ndarray]]:
    """Retrieve pages from cache if not expired."""
    _cleanup_expired_cache()
    entry = _page_cache.get(doc_id)
    if entry:
        timestamp, pages = entry
        if time.time() - timestamp <= _CACHE_TTL_SECONDS:
            return pages
        else:
            del _page_cache[doc_id]  # Remove expired entry
    return None


def _cache_pages(doc_id: str, pages: List[np.ndarray]):
    """Store pages in cache with timestamp."""
    _page_cache[doc_id] = (time.time(), pages)


def _compute_doc_id(filename: str, content: bytes) -> str:
    digest = hashlib.sha256((filename + content[:8192].hex()).encode()).hexdigest()
    return digest[:24]


def _validate_file(filename: str, content: bytes, mime_type: str) -> None:
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise FileTooLargeError(size_mb, settings.MAX_UPLOAD_SIZE_MB)
    if mime_type not in settings.SUPPORTED_MIME_TYPES:
        raise UnsupportedFileTypeError(mime_type)


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> List[np.ndarray]:
    try:
        from pdf2image import convert_from_bytes
    except ImportError as exc:
        raise PDFConversionError("pdf2image not installed") from exc
    try:
        return [np.array(img) for img in convert_from_bytes(pdf_bytes, dpi=dpi, fmt="RGB")]
    except Exception as exc:
        raise PDFConversionError(f"PDF conversion failed: {exc}") from exc


def _image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    try:
        from PIL import Image
        import io
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception as exc:
        raise PDFConversionError(f"Image decoding failed: {exc}") from exc


def _load_pages(content: bytes, mime_type: str) -> List[np.ndarray]:
    if mime_type == "application/pdf":
        return _pdf_to_images(content)
    return [_image_bytes_to_array(content)]


class DocumentProcessor:
    """
    Orchestrates the full DocuParse pipeline with Supabase persistence.

    All public methods are async and accept a SQLAlchemy AsyncSession
    injected via FastAPI's Depends mechanism.
    """

    def __init__(self):
        self._ocr = OCRPipeline.get_instance()
        self._layout = LayoutAnalyser()
        self._layoutlmv3 = get_layoutlmv3_scorer()

    async def upload(
        self,
        session: AsyncSession,
        filename: str,
        content: bytes,
        mime_type: str,
    ) -> UploadResponse:
        """Validate, convert, and register an uploaded document."""
        _validate_file(filename, content, mime_type)
        doc_id = _compute_doc_id(filename, content)

        existing = await repo.get_document(session, doc_id)
        if existing:
            logger.info("Document already uploaded: %s", doc_id)
            return UploadResponse(
                document_id=doc_id,
                filename=existing.filename,
                size_bytes=existing.size_bytes,
                mime_type=existing.mime_type,
                total_pages=existing.total_pages,
                message="Already uploaded. Use /parse to run the OCR pipeline.",
            )

        pages = _load_pages(content, mime_type)
        total_pages = len(pages)

        await repo.create_document(
            session, doc_id=doc_id, filename=filename,
            mime_type=mime_type, size_bytes=len(content), total_pages=total_pages,
        )
        await repo.create_parse_job(session, document_id=doc_id)
        _cache_pages(doc_id, pages)

        logger.info("Uploaded '%s' -> %s (%d pages)", filename, doc_id, total_pages)
        return UploadResponse(
            document_id=doc_id, filename=filename,
            size_bytes=len(content), mime_type=mime_type, total_pages=total_pages,
        )

    async def parse(self, session: AsyncSession, doc_id: str) -> ParseResponse:
        """Run the full pipeline. Returns Supabase-cached result if available."""
        doc = await repo.get_document(session, doc_id)
        if doc is None:
            raise DocumentNotFoundError(doc_id)

        # Return Supabase-cached result (survives restarts)
        cached = await repo.get_cached_result(session, doc_id)
        if cached:
            logger.info("Returning Supabase-cached result for %s", doc_id)
            return ParseResponse(**cached)

        pages = _get_cached_pages(doc_id)
        if not pages:
            raise DocumentNotFoundError(doc_id + " (re-upload required after restart)")

        await repo.set_job_processing(session, doc_id)
        total_start = time.perf_counter()
        page_results: List[PageResult] = []

        page_results: List[PageResult] = []

        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            async def process_page(page_num: int, page_image: np.ndarray) -> PageResult:
                """Process a single page asynchronously."""
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    # Run OCR in thread pool (CPU-bound)
                    raw_output = await loop.run_in_executor(executor, self._ocr.run, page_image, page_num)
                    
                    # Layout analysis (also CPU-bound)
                    page_result = await loop.run_in_executor(executor, self._layout.analyse, raw_output, page_num)

                    # LayoutLMv3 re-scoring (GPU/CPU-bound)
                    if self._layoutlmv3.is_available and page_result.regions:
                        rescored = await loop.run_in_executor(
                            executor, self._layoutlmv3.score_regions,
                            page_result.regions, page_image, 0.70
                        )
                        page_result = page_result.model_copy(update={"regions": rescored})

                return page_result

            # Process pages in parallel
            tasks = [process_page(page_num, page_image) for page_num, page_image in enumerate(pages, start=1)]
            page_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for i, result in enumerate(page_results):
                if isinstance(result, Exception):
                    logger.error("Page %d processing failed: %s", i+1, result)
                    raise result
                page_results[i] = result

        except Exception as exc:
            await repo.set_job_failed(session, doc_id, str(exc))
            raise

        total_elapsed = round(time.perf_counter() - total_start, 4)

        response = ParseResponse(
            document_id=doc_id,
            filename=doc.filename,
            total_pages=doc.total_pages,
            pages=page_results,
            processing_time_seconds=total_elapsed,
        )

        # Persist to Supabase JSONB
        await repo.set_job_complete(
            session, document_id=doc_id,
            result=response.model_dump(), processing_time_seconds=total_elapsed,
        )

        logger.info("Parse complete for '%s' in %.3fs", doc.filename, total_elapsed)
        return response

    async def get_record(self, session: AsyncSession, doc_id: str) -> Optional[dict]:
        doc = await repo.get_document(session, doc_id)
        if doc is None:
            return None
        job = await repo.get_parse_job(session, doc_id)
        return {
            "filename": doc.filename,
            "mime_type": doc.mime_type,
            "size_bytes": doc.size_bytes,
            "total_pages": doc.total_pages,
            "is_parsed": doc.is_parsed,
            "parse_status": job.status if job else "no_job",
            "parsed": job.result if (job and job.status == "complete") else None,
        }

    async def invalidate(self, session: AsyncSession, doc_id: str) -> bool:
        deleted = await repo.delete_document(session, doc_id)
        _page_cache.pop(doc_id, None)
        return deleted
