"""
Database Layer — Supabase / PostgreSQL
=======================================
Replaces the in-memory document store with a persistent Supabase-backed
PostgreSQL store via SQLAlchemy (async).

Schema:
    documents   — uploaded file metadata + page count
    parse_jobs  — pipeline execution status + cached result (JSONB)

Using SQLAlchemy rather than the Supabase Python SDK directly gives us:
  - Async support via asyncpg
  - Full ORM for future migrations
  - Database-agnostic queries (easy to swap to local Postgres in dev)

The Supabase connection string is:
    postgresql+asyncpg://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres

Set SUPABASE_DB_URL in .env — get it from:
    Supabase dashboard → Project Settings → Database → Connection String → URI (Transaction mode)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, Index, Integer,
    String, Text, UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """
    Stores uploaded document metadata.
    The raw page images are NOT stored (too large for DB); they're held in
    process memory during the parse job and discarded after.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    is_parsed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index("ix_documents_created_at", "created_at"),
    )


class ParseJobRecord(Base):
    """
    Stores the cached parse result as JSONB.

    JSONB is chosen over TEXT for the result payload so that Supabase's
    dashboard and downstream queries can filter/query inside the document
    structure (e.g., "find all documents with a table on page 1").
    """
    __tablename__ = "parse_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending | processing | complete | failed
    result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_parse_jobs_document_id", "document_id"),
        UniqueConstraint("document_id", name="uq_parse_jobs_document_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine & session factory
# ─────────────────────────────────────────────────────────────────────────────

def get_engine():
    global _engine
    if _engine is None:
        if not settings.SUPABASE_DB_URL:
            raise RuntimeError(
                "SUPABASE_DB_URL not set. "
                "Add it to .env: postgresql+asyncpg://postgres.[ref]:[pw]@..."
            )
        _engine = create_async_engine(
            settings.SUPABASE_DB_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,       # Detect stale connections
            pool_recycle=1800,        # Recycle connections every 30 min
            echo=settings.DEBUG,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables():
    """Create all tables if they don't exist (run on startup)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured.")


async def check_connection() -> bool:
    """Health check — verify the database is reachable."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False
