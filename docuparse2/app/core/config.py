"""
Application configuration — all settings sourced from environment variables
with sensible defaults for local development.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Service ──────────────────────────────────────────────────────────────
    APP_NAME: str = "DocuParse"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── File handling ─────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "/tmp/docuparse_uploads"
    SUPPORTED_MIME_TYPES: List[str] = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    ]

    # ── OCR Pipeline (PP-OCR stage config) ────────────────────────────────────
    # Stage 1: Text detection
    OCR_DET_ALGORITHM: str = "DB"          # Differentiable Binarization
    OCR_DET_DB_THRESH: float = 0.3
    OCR_DET_DB_BOX_THRESH: float = 0.5
    OCR_DET_DB_UNCLIP_RATIO: float = 1.6  # Controls bounding box expansion

    # Stage 2: Direction classification
    OCR_USE_ANGLE_CLS: bool = True
    OCR_CLS_THRESH: float = 0.9

    # Stage 3: Text recognition
    OCR_REC_ALGORITHM: str = "SVTR_LCNet"
    OCR_LANG: str = "en"
    OCR_USE_GPU: bool = False

    # ── Layout analysis (LayoutLM-inspired) ───────────────────────────────────
    # DBSCAN clustering parameters
    DBSCAN_EPS_Y: float = 15.0        # Vertical proximity threshold (pixels)
    DBSCAN_EPS_X: float = 50.0        # Horizontal proximity threshold (pixels)
    DBSCAN_MIN_SAMPLES: int = 1

    # Heuristic thresholds for region classification
    HEADING_HEIGHT_RATIO: float = 1.4  # Bounding box height ratio vs median
    TABLE_ALIGNMENT_TOLERANCE: float = 10.0  # Pixels for column alignment detection
    FIGURE_ASPECT_RATIO_MIN: float = 0.5
    FIGURE_MIN_AREA_PX: int = 10000

    # ── Supabase / PostgreSQL ─────────────────────────────────────────────────
    SUPABASE_DB_URL: str = ""
    # Format: postgresql+asyncpg://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres

    # ── LayoutLMv3 ────────────────────────────────────────────────────────────
    LAYOUTLMV3_ENABLED: bool = False           # Enable LayoutLMv3 re-scoring
    LAYOUTLMV3_RESCORE_THRESHOLD: float = 0.70 # Rescore regions below this confidence
    LAYOUTLMV3_CHECKPOINT: str = "nielsr/layoutlmv3-finetuned-publaynet"

    # ── Redis (optional L1 cache in front of Supabase) ────────────────────────
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
