"""
Document Repository
====================
All database read/write operations for documents and parse jobs.
Follows the Repository pattern: the rest of the application never
touches SQLAlchemy directly — it calls these functions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentRecord, ParseJobRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Document operations
# ─────────────────────────────────────────────────────────────────────────────

async def create_document(
    session: AsyncSession,
    doc_id: str,
    filename: str,
    mime_type: str,
    size_bytes: int,
    total_pages: int,
) -> DocumentRecord:
    record = DocumentRecord(
        id=doc_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        total_pages=total_pages,
    )
    session.add(record)
    await session.flush()
    logger.info("Created document record: %s ('%s')", doc_id, filename)
    return record


async def get_document(session: AsyncSession, doc_id: str) -> Optional[DocumentRecord]:
    result = await session.execute(select(DocumentRecord).where(DocumentRecord.id == doc_id))
    return result.scalar_one_or_none()


async def mark_document_parsed(session: AsyncSession, doc_id: str) -> None:
    await session.execute(
        update(DocumentRecord)
        .where(DocumentRecord.id == doc_id)
        .values(is_parsed=True)
    )


async def delete_document(session: AsyncSession, doc_id: str) -> bool:
    record = await get_document(session, doc_id)
    if record is None:
        return False
    await session.delete(record)
    # Cascade delete handled at DB level; also clean up parse job
    job = await get_parse_job(session, doc_id)
    if job:
        await session.delete(job)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Parse job operations
# ─────────────────────────────────────────────────────────────────────────────

async def create_parse_job(
    session: AsyncSession,
    document_id: str,
) -> ParseJobRecord:
    job = ParseJobRecord(document_id=document_id, status="pending")
    session.add(job)
    await session.flush()
    return job


async def get_parse_job(
    session: AsyncSession,
    document_id: str,
) -> Optional[ParseJobRecord]:
    result = await session.execute(
        select(ParseJobRecord).where(ParseJobRecord.document_id == document_id)
    )
    return result.scalar_one_or_none()


async def set_job_processing(session: AsyncSession, document_id: str) -> None:
    await session.execute(
        update(ParseJobRecord)
        .where(ParseJobRecord.document_id == document_id)
        .values(status="processing")
    )


async def set_job_complete(
    session: AsyncSession,
    document_id: str,
    result: dict,
    processing_time_seconds: float,
) -> None:
    await session.execute(
        update(ParseJobRecord)
        .where(ParseJobRecord.document_id == document_id)
        .values(
            status="complete",
            result=result,
            processing_time_seconds=processing_time_seconds,
            completed_at=datetime.now(timezone.utc),
        )
    )
    await mark_document_parsed(session, document_id)


async def set_job_failed(
    session: AsyncSession,
    document_id: str,
    error_message: str,
) -> None:
    await session.execute(
        update(ParseJobRecord)
        .where(ParseJobRecord.document_id == document_id)
        .values(
            status="failed",
            error_message=error_message,
            completed_at=datetime.now(timezone.utc),
        )
    )


async def get_cached_result(
    session: AsyncSession,
    document_id: str,
) -> Optional[dict]:
    """Return the cached JSONB parse result, or None if not yet parsed."""
    job = await get_parse_job(session, document_id)
    if job and job.status == "complete" and job.result:
        return job.result
    return None
