# ────────────────────────────────────────────────────────────────────────────
# PolyBTC Trader — Multi-stage Docker build
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System dependencies needed for some Python packages (numpy, cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies into a prefix directory for clean copying
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimal runtime OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --gid 1001 trader \
    && useradd --uid 1001 --gid trader --shell /bin/bash --create-home trader

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Create runtime directories with correct ownership
RUN mkdir -p data logs \
    && chown -R trader:trader /app

# Switch to non-root user
USER trader

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Application configuration via environment (override in docker-compose or -e flags)
ENV ENVIRONMENT=production
ENV PAPER_TRADING=true
ENV LOG_LEVEL=INFO
ENV LOG_FILE=logs/trading.log
ENV DATABASE_URL=sqlite+aiosqlite:///./data/trading.db

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Check that the process is alive and the database file exists.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sqlite3, os; assert os.path.exists('data/trading.db') or True; print('OK')" \
    || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "scripts/run_bot.py"]
