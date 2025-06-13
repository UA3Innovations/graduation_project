# Schedule Optimization Pipeline

## Overview

This project implements a comprehensive **Schedule Optimization Pipeline** that integrates four key components:

1. **Bus Simulation Pipeline** - Generates realistic passenger flow data
2. **Bus Optimization Pipeline** - Uses genetic algorithms to optimize bus schedules
3. **Bus Prediction Pipeline** - Employs LSTM+Prophet hybrid models for passenger flow prediction
4. **Bus Evaluation Pipeline** - Evaluates optimization performance with comprehensive metrics

## Architecture

The pipeline follows this execution flow:
```
Simulation → Optimization → Prediction → Evaluation
```

### Key Features

- **Genetic Algorithm Optimization**: Optimizes bus schedules for all lines simultaneously
- **Hybrid ML Prediction**: Combines LSTM and Prophet models for accurate passenger flow prediction
- **Realistic Constraints**: Implements occupancy limits, last-stop logic, and night-time constraints
- **Comprehensive Evaluation**: Multi-metric evaluation including passenger conservation, resource constraints, and statistical analysis
- **Docker Support**: Fully containerized for easy deployment
- **Configuration Management**: Quick/Full modes with different optimization parameters

## Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- Required packages: `pip install -r requirements.txt`

### Running the Pipeline

```bash
# Quick mode (30 generations, faster execution)
python -m main_pipeline.integrated_pipeline --mode quick

# Full mode (150 generations, comprehensive optimization)
python -m main_pipeline.integrated_pipeline --mode full
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Project Structure

```
├── bus_simulation_pipeline/     # Passenger flow simulation
├── bus_optimization_pipeline/   # Genetic algorithm optimization
├── bus_prediction_pipeline/     # LSTM+Prophet prediction models
├── bus_evaluation_pipeline/     # Performance evaluation
├── main_pipeline/              # Integrated pipeline orchestration
├── config/                     # Configuration files
├── docker/                     # Docker configuration
└── outputs/                    # Pipeline results
```

## Configuration

The pipeline supports two execution modes:

- **Quick Mode**: 30 generations, 30 population size, 50 LSTM epochs
- **Full Mode**: 150 generations, 80 population size, 100 LSTM epochs

See `CONFIGURATION_GUIDE.md` for detailed configuration options.

## Results

The pipeline generates comprehensive outputs including:

- Optimized bus schedules for all lines
- Passenger flow predictions
- Performance evaluation metrics
- Visualization plots
- Detailed analysis reports

## Recent Improvements

- ✅ Fixed occupancy rate constraints (max 200% capacity)
- ✅ Improved genetic algorithm fitness function balance
- ✅ Enhanced service coverage optimization
- ✅ Added comprehensive constraint validation
- ✅ Implemented realistic passenger flow generation

## Contributing

This project is part of a graduation thesis on bus schedule optimization using machine learning and genetic algorithms.

## License

Academic use only - Part of graduation project at [University Name]. 