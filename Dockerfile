# Financial AI Lending Platform - Dockerfile
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including Git LFS
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    git-lfs \
    curl \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements-prod.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Initialize Git LFS and pull large files
RUN git lfs install && \
    if [ -f .gitattributes ]; then \
        echo "Git LFS attributes found, pulling LFS files..."; \
        git lfs pull || echo "Git LFS pull failed - files may already be present"; \
    else \
        echo "No .gitattributes found, skipping LFS pull"; \
    fi

# Create necessary directories
RUN mkdir -p data/actual data/outputs data/models logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command - run the main dashboard
CMD ["streamlit", "run", "src/executive_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
