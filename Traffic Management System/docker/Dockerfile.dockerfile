# docker/Dockerfile
# =============================================================================
# Smart Traffic Management System - API Container
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY src/api/requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Install production server
RUN pip install --no-cache-dir gunicorn

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="mhdbel@example.com" \
      version="1.0.0" \
      description="Smart Traffic Management System API" \
      org.opencontainers.image.source="https://github.com/mhdbel/Smart-Traffic-Management-System"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Application settings
    PORT=5000 \
    HOST=0.0.0.0 \
    WORKERS=4 \
    TIMEOUT=120 \
    FLASK_ENV=production \
    LOG_LEVEL=INFO

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup src/api/ ./api/
COPY --chown=appuser:appgroup src/agents/ ./agents/

# Copy models if they exist
COPY --chown=appuser:appgroup models/ ./models/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Graceful shutdown
STOPSIGNAL SIGTERM

# Default command - production server
CMD ["sh", "-c", "gunicorn --bind ${HOST}:${PORT} --workers ${WORKERS} --timeout ${TIMEOUT} --access-logfile - --error-logfile - --capture-output api.app:app"]
