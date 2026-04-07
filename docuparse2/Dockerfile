# ─────────────────────────────────────────────────────────────────────────────
# DocuParse — Production Dockerfile
# Multi-stage build: deps layer cached separately from app layer so that
# source-only changes don't trigger a full pip install.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies required by PaddleOCR and pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \      
    # pdf2image depends on poppler
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate prefix for clean copying
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="DocuParse <docuparse@example.com>"
LABEL description="Layout-aware document intelligence API (PP-OCR + DBSCAN)"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd -r docuparse && useradd -r -g docuparse docuparse

# Runtime system deps (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY app/ ./app/
COPY tests/ ./tests/

# PaddleOCR downloads model weights to ~/.paddleocr on first use.
# Pre-creating the directory and making it writable avoids permission errors.
RUN mkdir -p /home/docuparse/.paddleocr && \
    chown -R docuparse:docuparse /home/docuparse /app

USER docuparse

# Environment defaults (overridden by docker-compose / K8s env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    OCR_USE_GPU=false \
    REDIS_ENABLED=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use Gunicorn with Uvicorn workers for production; override CMD for dev.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
