# Loan Prediction System - Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including Git LFS (lighter than main app - no GIS libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Initialize Git LFS and pull large files (if any in this service)
RUN git lfs install && \
    if [ -f .gitattributes ]; then \
        echo "Git LFS attributes found, pulling LFS files..."; \
        git lfs pull || echo "Git LFS pull failed - files may already be present"; \
    else \
        echo "No .gitattributes found, skipping LFS pull"; \
    fi

# Create necessary directories
RUN mkdir -p data models logs config


# Expose Streamlit port
EXPOSE 8502

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# Default command - run the loan prediction dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]
