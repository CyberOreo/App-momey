# PolyBTC Trader — Cloud Deployment
# Runs the web dashboard (web_app.py) which can launch the 5-min engine
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements_core.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_core.txt

# Copy application
COPY web_app.py .
COPY static/ ./static/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

RUN mkdir -p data logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PAPER_TRADING=true
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/status', timeout=4)" \
    || exit 1

CMD ["python", "web_app.py"]
