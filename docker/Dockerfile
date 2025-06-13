# Multi-stage Dockerfile for Schedule Optimization Project
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from all pipelines
COPY bus_simulation_pipeline/requirements.txt ./requirements_simulation.txt
COPY bus_optimization_pipeline/requirements.txt ./requirements_optimization.txt
COPY bus_prediction_pipeline/requirements.txt ./requirements_prediction.txt
COPY bus_evaluation_pipeline/requirements.txt ./requirements_evaluation.txt

# Create combined requirements file
RUN cat requirements_*.txt | sort | uniq > requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy all pipeline packages
COPY bus_simulation_pipeline/ ./bus_simulation_pipeline/
COPY bus_optimization_pipeline/ ./bus_optimization_pipeline/
COPY bus_prediction_pipeline/ ./bus_prediction_pipeline/
COPY bus_evaluation_pipeline/ ./bus_evaluation_pipeline/

# Copy main pipeline
COPY main_pipeline/ ./main_pipeline/
COPY config/ ./config/
COPY data/ ./data/

# Create output directories
RUN mkdir -p outputs logs

# Set permissions
RUN chmod +x main_pipeline/integrated_pipeline.py

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "main_pipeline/integrated_pipeline.py", "--mode", "quick"] 