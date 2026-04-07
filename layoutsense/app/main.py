"""
LayoutSense — Layout-Aware Document Intelligence API
===================================================
A 3-stage OCR pipeline inspired by PP-OCR (Du et al., 2020) with
LayoutLM-style spatial reasoning for document structure reconstruction.
"""

import logging
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    logger.info("LayoutSense starting up...")

    # Initialise Supabase database tables
    if settings.SUPABASE_DB_URL:
        try:
            from app.db.models import create_tables, check_connection
            await create_tables()
            ok = await check_connection()
            logger.info("Supabase connected: %s", ok)
        except Exception as exc:
            logger.error("Supabase init failed: %s", exc)
    else:
        logger.warning("SUPABASE_DB_URL not set — database persistence disabled.")

    # Warm up the OCR pipeline on startup to avoid cold-start latency
    try:
        from app.services.ocr_pipeline import OCRPipeline
        pipeline = OCRPipeline.get_instance()
        logger.info("OCR pipeline initialized successfully.")
    except Exception as exc:
        logger.warning(f"OCR pipeline warm-up failed (non-fatal): {exc}")

    # Initialize Redis connection pool if enabled
    if settings.REDIS_ENABLED:
        try:
            app.state.redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
            )
            await app.state.redis.ping()
            logger.info(f"Redis connected at {settings.REDIS_URL}")
        except Exception as exc:
            logger.warning(f"Redis connection failed (caching disabled): {exc}")
            app.state.redis = None
    else:
        app.state.redis = None

    yield  # Application running

    # Graceful shutdown
    logger.info("LayoutSense shutting down...")
    if getattr(app.state, "redis", None):
        await app.state.redis.aclose()


app = FastAPI(
    title="LayoutSense",
    description=(
        "Layout-aware document intelligence API implementing a PP-OCR-inspired "
        "3-stage pipeline with LayoutLM-style spatial clustering for document "
        "structure reconstruction."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting middleware
# NOTE: Rate limiting is currently disabled due to incompatibility with slowapi and UploadFile
# A middleware-based approach (e.g., per-IP limits in reverse proxy) is recommended for production
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Attach X-Process-Time header to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "type": type(exc).__name__},
    )


app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe — confirms the service is reachable."""
    return {"status": "healthy", "service": "LayoutSense", "version": "1.0.0"}


@app.get("/ready", tags=["Health"])
async def readiness_check(request: Request):
    """Readiness probe — confirms critical dependencies are available."""
    checks = {"ocr_pipeline": False, "redis": False}

    try:
        from app.services.ocr_pipeline import OCRPipeline
        OCRPipeline.get_instance()
        checks["ocr_pipeline"] = True
    except Exception:
        pass

    if getattr(request.app.state, "redis", None):
        try:
            await request.app.state.redis.ping()
            checks["redis"] = True
        except Exception:
            pass
    else:
        checks["redis"] = "disabled"

    all_critical_ready = checks["ocr_pipeline"]
    return JSONResponse(
        status_code=200 if all_critical_ready else 503,
        content={"status": "ready" if all_critical_ready else "degraded", "checks": checks},
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Basic metrics for monitoring (JSON format)."""
    import psutil
    import time
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        "timestamp": int(time.time()),
        "service": "LayoutSense",
        "version": "1.0.0",
        "memory_mb": round(memory_mb, 2),
        "cpu_percent": process.cpu_percent(),
        "uptime_seconds": int(time.time() - process.create_time()),
    }
