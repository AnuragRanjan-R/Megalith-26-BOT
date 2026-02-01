# ============================================
# Chatbot Backend Dockerfile (Optimized)
# ============================================

# ----------- BUILD STAGE -----------
# Use full Debian image to avoid heavy compilation issues
FROM python:3.11-bookworm AS builder

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System packages required to build some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies using prebuilt wheels
RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt


# ----------- PRODUCTION STAGE -----------
FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime libraries needed for numpy, torch, chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libstdc++6 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Prepare app directories
RUN mkdir -p chroma_db data && chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

<<<<<<< HEAD
=======
# Start the application
>>>>>>> 45ce9e48f33ec719aa9d95ffc155f4ac0886241d
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
