# Multi-stage build for efficient Azure deployment
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY pipeline/ pipeline/
COPY config/ config/
COPY scripts/ scripts/
COPY data/ data/

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV PYTHONPATH=/app/src:/app/pipeline
ENV PYTHONUNBUFFERED=1

# Azure-specific configurations
ENV AZURE_STORAGE_CONNECTION_STRING=""
ENV AZURE_STORAGE_ACCOUNT=""
ENV AZURE_STORAGE_CONTAINER=""

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "pipeline/pipeline_runner.py"]

# Production stage
FROM base as production

# Copy only necessary files for production
COPY --from=base /app /app

# Set production environment
ENV ENVIRONMENT=production

# Override entrypoint for production
ENTRYPOINT ["python", "pipeline/pipeline_runner.py"] 