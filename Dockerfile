# =============================================================================
# Stage 1: Builder — Install all Python dependencies
# =============================================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps for OCR (tesseract + poppler) and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpoppler-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime — Lean production image
# =============================================================================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy system libs needed at runtime (tesseract, poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY main.py .
COPY .env .

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8888

# Default: run API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "2"]
