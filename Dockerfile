# Multi-stage Dockerfile for aria-core API
# Stage 1: Build dependencies
# Stage 2: Production runtime

# ============================================================
# Stage 1 — Builder
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --no-cache-dir hatchling

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# Install all optional deps as wheels
RUN pip wheel --wheel-dir /wheels \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "python-jose[cryptography]>=3.3.0" \
    "websockets>=12.0" \
    "sqlalchemy[asyncio]>=2.0.25" \
    "asyncpg>=0.29.0" \
    "alembic>=1.13.0"

# ============================================================
# Stage 2 — Production
# ============================================================
FROM python:3.12-slim AS production

# Security: non-root user
RUN groupadd -r aria && useradd -r -g aria -d /app -s /sbin/nologin aria

WORKDIR /app

# Install from pre-built wheels (no compile needed)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

# Copy alembic config
COPY alembic.ini ./
COPY src/aria_core/persistence/postgres/migrations/ src/aria_core/persistence/postgres/migrations/

# Switch to non-root user
USER aria

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Production uvicorn with workers
CMD ["uvicorn", "aria_core.api:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log"]
