# Multi-stage build for efficient local deployment
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY pipeline/ pipeline/
COPY config/ config/
COPY scripts/ scripts/
COPY data/ data/
COPY tests/ tests/

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV PYTHONPATH=/app/src:/app/pipeline:/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from core.data_models import BusTransitData; print('Simulation ready')"

# Expose port for potential web interface (future use)
EXPOSE 8080

# Default command - run the pipeline
CMD ["python", "scripts/run_simulation.py"] 