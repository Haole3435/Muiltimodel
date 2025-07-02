# Multi-stage build for A05 Knowledge Base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    # Dependencies for Playwright
    libnss3 \
    libnspr4 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    # Dependencies for image processing
    libopencv-dev \
    python3-opencv \
    # Dependencies for audio processing
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy source code
COPY src/ ./src/
COPY scrapers/ ./scrapers/
COPY inference/ ./inference/
COPY agents/ ./agents/
COPY vector_stores/ ./vector_stores/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/tmp

# Set up data directories
RUN mkdir -p /app/data/{raw,processed,embeddings,models} && \
    chmod -R 755 /app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables for optimization
ENV CRAWL4AI_CACHE_DIR=/app/cache
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# Start command
CMD ["python", "src/main.py"]

